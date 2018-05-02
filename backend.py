"""Backend for running ONNX on Keras

To run this, you will need to have Keras installed as well.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import collections
import re
import warnings
import sys
import itertools
from math import ceil, floor

try:
  from itertools import izip as zip
except ImportError: # will be 3.x series
  pass

import numpy as np
from onnx import checker
from onnx.onnx_pb2 import GraphProto, TensorProto, AttributeProto
from keras_net import KerasNet
# from onnx_tf.backend_rep import TensorflowRep
import onnx.numpy_helper
import onnx.defs
from keras.models import Model, Sequential
from keras.layers import Input, Lambda
import keras
import keras.backend as K


from onnx.backend.base import (
    Backend,
    BackendRep,
    Device,
    DeviceType,
    namedtupledict,
)

from onnx import onnx_pb2, helper

def convertAttributeProto(onnx_arg):
  """
  Convert an ONNX AttributeProto into an appropriate Python object
  for the type.
  NB: Tensor attribute gets returned as the straight proto.
  """
  if onnx_arg.HasField('f'):
    return onnx_arg.f
  elif onnx_arg.HasField('i'):
    return onnx_arg.i
  elif onnx_arg.HasField('s'):
    return str(onnx_arg.s, 'utf-8') \
      if sys.version_info[0] >= 3 else onnx_arg.s
  elif onnx_arg.HasField('t'):
    return onnx_arg.t  # this is a proto!
  elif onnx_arg.floats:
    return list(onnx_arg.floats)
  elif onnx_arg.ints:
    return list(onnx_arg.ints)
  elif onnx_arg.strings:
    str_list = list(onnx_arg.strings)
    if sys.version_info[0] >= 3:
      str_list = map(lambda x: str(x, 'utf-8'), str_list)
    return str_list
  else:
    raise ValueError("Unsupported ONNX attribute: {}".format(onnx_arg))

class OnnxAttributes(dict):
  """
  This is a more convenient way to work with ONNX/Caffe2 attributes
  that is not the protobuf representation.
  """
  @staticmethod
  def from_onnx(args):
    d = OnnxAttributes()
    for arg in args:
      d[arg.name] = convertAttributeProto(arg)
    return d

class OnnxNode(object):
  """
  Reimplementation of NodeProto from ONNX, but in a form
  more convenient to work with from Python.
  We may temporarily edit these nodes to get them into Caffe2 form,
  before actually translating into the Caffe2 protobuf, since this
  is easier than decomposing everything, and putting it back together
  when we're ready.
  """
  def __init__(self, node):
    self.name = str(node.name)
    self.op_type = str(node.op_type)
    self.attrs = OnnxAttributes.from_onnx(node.attribute)
    self.consumed_inputs = self.attrs.pop("consumed_inputs", None)
    self.inputs = list(node.input)
    self.outputs = list(node.output)
    self.node_proto = node

class KerasBackend(Backend):
  """ Tensorflow Backend for ONNX
  """
  extra_input = list()

  onnx_tf_attribute_map = {
      "scale": "stddev",
      "high": "maxval",
      "low": "minval",
      "axes": "axis",
      "axis": "dim",
      "to": "dtype",
  }

  onnx_keras_op_map = {
      "abs": K.abs,
      "cast": K.cast,
      "exp": K.exp,
      "hardsigmoid": K.hard_sigmoid,
      "log": K.log,
      "random_normal": K.random_normal,
      "random_uniform": K.random_uniform,
      "reduce_log_sum_exp": K.logsumexp,
      "reduce_max": K.max,
      "reduce_mean": K.mean,
      "reduce_min": K.min,
      "reduce_prod": K.prod,
      "reduce_sum": K.sum,
      "relu": K.relu,
      "sigmoid": K.sigmoid,
      "softplus": K.softplus,
      "softsign": K.softsign,
      "sqrt": K.sqrt,
      "squeeze": K.squeeze,
      "tanh": K.tanh,
      "transpose": K.transpose,
  }

  tensor_type_to_keras_type = {
      TensorProto.FLOAT: 'float32',
      TensorProto.UINT8: 'uint8',
      TensorProto.INT8: 'int8',
      TensorProto.UINT16: 'uint16',
      TensorProto.INT16: 'int16',
      TensorProto.INT32: 'int32',
      TensorProto.INT64: 'int64',
      TensorProto.BOOL: 'bool',
      TensorProto.FLOAT16: 'float16',
      TensorProto.DOUBLE: 'float64',
      TensorProto.COMPLEX64: 'complex64',
      TensorProto.COMPLEX128: 'complex128',
  }

  tensor_type_enum = [
      "undefined",
      'float32',
      'uint8',
      'int8',
      'uint16',
      'int16',
      'int32',
      'int64',
      'string',
      'bool',
      'float16',
      'float64',
      'complex64',
      'complex128',

  ]

  attr_translator = {
      "dtype": lambda cls, x: cls.tensor_type_to_keras_type[x],
      "keepdims": lambda cls, x: bool(x),
      "to": lambda cls, x: x,
  }

  @classmethod
  def get_keras_pad(cls, x, pads, dim, data_format=None):
    if len(pads) == dim*2:
      pads = list(np.transpose(np.array(pads).reshape([2, dim]).astype(np.int32)))
      pads = tuple(tuple(i) for i in pads)
    elif len(pads) == dim:
      pads = tuple((i, i) for i in pads)
    if dim == 1:
      return Lambda(lambda _x: K.temporal_padding(_x, pads))(x)
    elif dim == 2:
      return Lambda(lambda _x: K.spatial_2d_padding(_x, pads, data_format))(x)
    elif dim == 3:
      return Lambda(lambda _x: K.spatial_3d_padding(_x, pads, data_format))(x)
    else:
      raise NotImplementedError("padding with dim {} is not implemented.".format(dim))

  @classmethod
  def _explicit_broadcast(cls, tensor, broadcast_dim=1, total_num_dim=4):
    if not isinstance(broadcast_dim, list):
      broadcast_dim = [broadcast_dim]

    for i in range(total_num_dim):
      if i not in broadcast_dim:
        tensor = np.expand_dims(tensor, i)

    return tensor

  @classmethod
  def _bin_op(cls, node, input_dict, op_func, inputlist=True):
    x = input_dict[node.inputs[0]]
    y = input_dict[node.inputs[1]]
    tmp = np.array([0])

    if type(y) == type(tmp):
        y = keras.layers.Input(shape=y.shape)
        cls.extra_input.append(y)
    broadcast = node.attrs.get("broadcast", 1)
    if broadcast == 0:
      warnings.warn("Definition of {} with broadcast disabled is not "
                    "yet supported.".format(node.type), UserWarning)
    if "axis" in node.attrs.keys():
      num_ones_to_append = len(x.get_shape()) - \
                           len(y.get_shape()) - \
                           node.attrs["axis"] + 1
      if num_ones_to_append > 0:

        ones = np.ones([num_ones_to_append], 'int32')
        broadcasted_shape = np.concatenate([K.int_shape(y)[1:], ones], axis=0)
        y = keras.layers.Reshape(broadcasted_shape)(y)

    if inputlist:
      return op_func([x, y])
    else:
      return op_func(x, y)


  @classmethod
  def onnx_graph_to_keras_net(cls, graph_def):
    # initializer: TensorProtos representing the values to initialize
    # a given tensor.
    # initialized: A list of names of the initialized tensors.
    if graph_def.initializer:
      input_dict_items = cls.onnx_initializer_to_input_dict_items(
          graph_def.initializer)
      initialized = {init.name for init in graph_def.initializer}
    else:
      input_dict_items = []
      initialized = set()


    predict_net = KerasNet()
    predict_net.name = graph_def.name

    predict_net.external_input.extend(
        value_info.name for value_info in graph_def.input)
    predict_net.external_output.extend(
        value_info.name for value_info in graph_def.output)
    print(predict_net.external_output)
    # creating placeholders for currently unkown inputs
    for value_info in graph_def.input:
      if value_info.name in initialized:
        continue

      shape = list(d.dim_value for d in
                   value_info.type.tensor_type.shape.dim)
      x = Input(batch_shape=shape, name=value_info.name, dtype=
      cls.tensor_type_enum[value_info.type.tensor_type.elem_type])

      input_dict_items.append([value_info.name, x])

    # input dict: this dictionary is a map from variable names
    # to the latest produced tensors of the given name.
    # This dictionary will get updated as build the graph because
    # some ops may produce a result tensor with the same name as
    # the input tensor. The input dict tracks the latest produced
    # tensors.
    input_dict = dict(input_dict_items)
    # Since input dict may be updated, we need to keep a copy
    # of the original input dict where we track the earliest
    # defined tensors so we can have access to the placeholders
    # to feed in input tensors when we run the graph.
    original_input_dict = dict(input_dict_items)
    output_dict = dict()

    for node in graph_def.node:
      node = OnnxNode(node)

      output_ops = cls._onnx_node_to_keras_op(node, input_dict)
      curr_node_output_map = list(zip(node.outputs, output_ops))
      input_dict = dict(list(input_dict.items()) +
                        curr_node_output_map)

      output_dict = dict(list(output_dict.items()) +
                         curr_node_output_map)
      predict_net.op.extend(output_ops)

    predict_net.output_dict = output_dict
    return original_input_dict, predict_net

  @classmethod
  def prepare(cls, model, device='CPU', **kwargs):
    super(KerasBackend, cls).prepare(model, device, **kwargs)

    original_input_dict, predict_net = (
        cls.onnx_graph_to_keras_net(model.graph))

    initialized = {init.name for init in model.graph.initializer}
    uninitialized = [x for x in predict_net.external_input
                     if not x in initialized]

    inputs = [original_input_dict[a] for a in uninitialized]
    outputs = [predict_net.output_dict[a] for a in predict_net.external_output]

    res_model = Model(inputs=inputs+cls.extra_input,
                      outputs=outputs)
    print(res_model.layers)

  @classmethod
  def onnx_initializer_to_input_dict_items(cls,
                                           initializer,
                                           init_net_name='init'):

    def tensor2list(onnx_tensor):
      # Use the onnx.numpy_helper because the data may be raw
      return onnx.numpy_helper.to_array(onnx_tensor).tolist()
    input_dict = [(tp.name, np.array(tensor2list(tp))) for tp in initializer]

    return input_dict

  @classmethod
  def op_name_to_lower(cls, name):
    return re.sub('(?<!^)(?=[A-Z])', '_', name).lower()

  @classmethod
  def _onnx_node_to_keras_op(cls, node, input_dict):
    op_name_lowered = cls.op_name_to_lower(node.op_type)
    if op_name_lowered in cls.onnx_keras_op_map.keys():
      return cls.handle_trivial(node, input_dict)

    handler_name = "handle_" + op_name_lowered
    # Check if specialized handler exists.
    if handler_name in dir(cls):
      method_to_call = getattr(cls, handler_name)
      tmp = method_to_call(node, input_dict)
      # print('check', K.is_keras_tensor(tmp[0]), tmp[0])
      return tmp
    else:
      raise NotImplementedError("{} op is not implemented.".format(node.op_type))

  @classmethod
  def run_node(cls, node, inputs, uninit=[0]):
    super(KerasBackend, cls).run_node(node, inputs)
    node = OnnxNode(node)
    input_tensor = list()
    input_array = list()
    input_dict = dict()
    for i in range(len(inputs)):
        input_dict[node.inputs[i]] = inputs[i]
    for i in uninit:
        input_array.append(inputs[i])
        shape = list(inputs[i].shape)
        if len(shape) == 1:
            shape = [None, shape[0]]

        x = Input(batch_shape=shape, name=node.inputs[i], dtype=str(inputs[i].dtype))
        input_tensor.append(x)
        input_dict[node.inputs[i]] = x
    out = cls._onnx_node_to_keras_op(node, input_dict)[0]

    model = Model(inputs=input_tensor, outputs=out)
    if len(input_array) == 1:
        input_array = input_array[0]

    res = model.predict(input_array)
    return namedtupledict('Outputs', node.outputs)(*[res])

  @classmethod
  def handle_trivial(cls, node, input_dict):
    # Perform automatic attribute value translation.
    attrs = dict([(x, cls.attr_translator[x](cls, node.attrs[x]) \
      if x in cls.attr_translator else node.attrs[x]) \
      for x in node.attrs.keys()])

    # Create an identity map from onnx attribute names to tf
    # attribute names.
    attr_map = dict([(x, x) for x in node.attrs.keys()])

    # Modify the map accoridng to onnx_tf_attribute_map.
    attr_map = dict([(x, cls.onnx_tf_attribute_map[x] \
      if x in cls.onnx_tf_attribute_map.keys() else x) \
      for x in attr_map.keys()])

    # TODO: Per op attribute name mapping has the final say.

    # Substitute attribute names in attrs.
    attrs = dict([(attr_map[x], y) for (x, y) in attrs.items()])

    if 'axis' in attrs.keys():
        axis = attrs['axis']
        if isinstance(axis, list):
            print(attrs['axis'])
            # axis = np.array(axis).astype('int32')
            axis = [int(v) for v in axis]
            attrs['axis'] = axis
            print(attrs['axis'])
            print(type(attrs['axis'][0]))
    func = cls.onnx_keras_op_map[cls.op_name_to_lower(node.op_type)]

    inputs = input_dict[node.inputs[0]]
    res = Lambda(lambda a: func(a, *attrs))(inputs)
    return [res]

  @classmethod
  def handle_add(cls, node, input_dict):
    x = input_dict[node.inputs[0]]
    y = input_dict[node.inputs[1]]
    # tmp = np.array([0])
    #
    # if type(y) == type(tmp):
    #   total_num_dim = len(x.get_shape())
    #   y = cls._explicit_broadcast(y,0,total_num_dim-1)
    #   return [Lambda(lambda a: K.bias_add(a, K.constant(y)))(x)]

    return [cls._bin_op(node, input_dict, keras.layers.add)]

  @classmethod
  def handle_arg_max(cls, node, input_dict):
    data = input_dict[node.inputs[0]]
    axis = node.attrs["axis"]
    keepdims = node.attrs.get("keepdims", 1)
    if keepdims == 1:
      warnings.warn("Definition of ArgMax with keepdims enabled is "
                    "incompatible between onnx and keras.",
                    UserWarning)

    return [Lambda(lambda x: K.argmax(x, axis=axis))(data)]

  @classmethod
  def handle_arg_min(cls, node, input_dict):
    data = input_dict[node.inputs[0]]
    axis = node.attrs["axis"]
    keepdims = node.attrs.get("keepdims", 1)
    if keepdims == 1:
      warnings.warn("Definition of ArgMin with keepdims enabled is "
                    "incompatible between onnx and keras.",
                    UserWarning)
    return [Lambda(lambda x: K.argmin(x, axis=axis))(data)]

  @classmethod
  def _pool(cls, node, input_dict, pool_layer):

    x = input_dict[node.inputs[0]]
    x_rank = len(x.get_shape())

    # support_cuda = cls.supports_device("CUDA")
    # storage_format, compute_format = cls.get_data_format(x_rank, support_cuda)

    kernel_shape = node.attrs["kernel_shape"]
    kernel_shape = tuple(kernel_shape)
    strides = node.attrs["strides"]

    data_format = "channels_first"

    if "auto_pad" in node.attrs.keys():
        if node.attrs['auto_pad'] == 'SAME_UPPER':
            return [pool_layer(pool_size=kernel_shape, strides=strides,
                              padding='same', data_format=data_format)(x)]
        elif node.attrs['auto_pad'] == 'SAME_LOWER':
            raise NotImplementedError('same_lower auto_pad is not implemented')

    if "pads" in node.attrs.keys():
        x = cls.get_keras_pad(x, node.attrs["pads"], x_rank-2, data_format)

    pooled = pool_layer(pool_size=kernel_shape, strides=strides, data_format=data_format)(x)

    return [pooled]

  @classmethod
  def handle_average_pool(cls, node, input_dict):
      x = input_dict[node.inputs[0]]
      dim = K.ndim(x) - 2
      if dim == 1:
          pool = keras.layers.AveragePooling1D
      elif dim == 2:
          pool = keras.layers.AveragePooling2D
      elif dim == 3:
          pool = keras.layers.AveragePooling3D
      else:
          raise NotImplementedError('max pooling with dim {} is not implemented.'.format(dim))
      return cls._pool(node, input_dict, pool)

  @classmethod
  def handle_batch_normalization(cls, node, input_dict):
    x = input_dict[node.inputs[0]]
    scale = input_dict[node.inputs[1]]
    bias = input_dict[node.inputs[2]]
    mean = input_dict[node.inputs[3]]
    variance = input_dict[node.inputs[4]]

    epsilon = node.attrs.get("epsilon", 0.00001)
    momentum = node.attrs.get("momentum", 0.9)

    # TODO is_test attr
    if "spatial" in node.attrs.keys():
      warnings.warn("Unsupported spatial attribute by keras in "
                    "batch_normalization. This attribute will be ignored.",
                    UserWarning)

    print('bn', K.int_shape(x), scale.shape, bias.shape, mean.shape, variance.shape)
    return [keras.layers.BatchNormalization(axis=1, momentum=momentum, epsilon=epsilon,
                                            weights=[scale, bias, mean, variance])(x)]
                                            # weights=[bias, variance, mean, scale])(x)]
  @classmethod
  def handle_clip(cls, node, input_dict):
    assert "max" in node.attrs.keys()
    assert "min" in node.attrs.keys()

    max_val = node.attrs["max"]
    min_val = node.attrs["min"]

    return [Lambda(lambda x: K.clip(x, min_val, max_val))(input_dict[node.inputs[0]])]

  @classmethod
  def handle_concat(cls, node, input_dict):
    values = [input_dict[a] for a in node.inputs]
    axis = node.attrs.get("axis", 1)

    return [keras.layers.concatenate(values, axis=axis)]

  @classmethod
  def _conv(cls, node, input_dict, transpose=False):
    x = input_dict[node.inputs[0]]

    x_rank = len(x.get_shape())

    dim = x_rank - 2
    if dim > 4 or dim < 1:
      raise NotImplementedError('conv of dim {} is not implemented.'.format((dim)))
    data_format = 'channels_first'

    W_weights = input_dict[node.inputs[1]]

    filters = len(W_weights)
    W_weights = np.transpose(W_weights, [2, 3, 1, 0])
    dilations = node.attrs.get("dilations", [1]*dim)
    strides = node.attrs.get("strides", [1]*dim)
    kernel_size = node.attrs.get("kernel_shape")
    if "group" in node.attrs and node.attrs['group'] != 1:
      raise NotImplementedError("'group' attribute in conv is not implemented")

    padding='valid'

    if "auto_pad" in node.attrs.keys():
        if node.attrs['auto_pad'] == 'SAME_UPPER':
            padding = 'same'
        elif node.attrs['auto_pad'] == 'SAME_LOWER':
            raise NotImplementedError('same_lower auto_pad is not implemented')

    if "pads" in node.attrs.keys():
      x = cls.get_keras_pad(x, node.attrs["pads"], dim, data_format)

    if len(node.inputs) == 2:
      convolved = keras.layers.convolutional._Conv(
        rank=dim, filters=filters, kernel_size=kernel_size, data_format=data_format, padding=padding,
        dilation_rate=dilations, strides=strides, use_bias=False, weights=[W_weights])(x)
    else:
      bias = input_dict[node.inputs[2]]
      convolved = keras.layers.convolutional._Conv(
        rank=dim, filters=filters, kernel_size=kernel_size, data_format=data_format, padding=padding,
        dilation_rate=dilations, strides=strides, use_bias=True, weights=[W_weights, bias])(x)

    return [convolved]


  @classmethod
  def handle_conv(cls, node, input_dict):
    return cls._conv(node, input_dict)

  @classmethod
  def handle_dropout(cls, node, input_dict):
    x = input_dict[node.inputs[0]]
    is_test = node.attrs["is_test"] if "is_test" in node.attrs.keys() else 0
    if is_test:
      return [x]
    ratio = node.attrs["ratio"] if "ratio" in node.attrs.keys() else 0.5
    return [Lambda(lambda a: K.dropout(a, ratio))(x)]  # ratio or 1-ratio?

  @classmethod
  def handle_elu(cls, node, input_dict):
    x = input_dict[node.inputs[0]]

    alpha = node.attrs.get("alpha", 1.0)

    return [keras.layers.advanced_activations.ELU(alpha)(x)]

  @classmethod
  def handle_equal(cls, node, input_dict):
    # TODO attr broadcast
    return [cls._bin_op(node, input_dict, Lambda(lambda x: K.equal(x[0], x[1])), inputlist=True)]

  @classmethod
  def handle_greater(cls, node, input_dict):
    return [cls._bin_op(node, input_dict, Lambda(lambda x, y: K.greater(x, y)), inputlist=False)]

  @classmethod
  def handle_less(cls, node, input_dict):
    return [cls._bin_op(node, input_dict, Lambda(lambda x, y: K.less(x, y)), inputlist=False)]

  @classmethod
  def handle_flatten(cls, node, input_dict):
    tensor = input_dict[node.inputs[0]]
    axis = node.attrs["axis"] if "axis" in node.attrs.keys() else 1
    shape = K.int_shape(tensor)[1:]

    axis -= 1
    split0, split1 = np.split(shape, [axis])
    if len(split0) == 0:
        split1 = np.prod(split1)
        output_shape = np.array([split1])
    else:
        split0 = np.prod(split0)
        split1 = np.prod(split1)
        output_shape = np.stack([split0, split1])
    return [keras.layers.core.Reshape(output_shape)(tensor)]

  @classmethod
  def handle_gather(cls, node, input_dict):
      x = input_dict[node.inputs[0]]

      y = input_dict[node.inputs[1]]
      axis = node.attrs.get(['axis'], 0)
      if axis != 0:
          raise NotImplementedError
      return [Lambda(lambda _x, _y: K.gather(_x, _y))(x, y)]


  @classmethod
  def handle_gemm(cls, node, input_dict):
    x = input_dict[node.inputs[0]]

    y = input_dict[node.inputs[1]]

    z = input_dict[node.inputs[2]]
    if "transA" in node.attrs.keys() and node.attrs["transA"] == 1:
      x = K.transpose(x)
    if "transB" in node.attrs.keys() and node.attrs["transB"] == 1:
      y = np.transpose(y)
    alpha = node.attrs["alpha"] if "alpha" in node.attrs.keys() else 1.0
    beta = node.attrs["beta"] if "beta" in node.attrs.keys() else 1.0

    layer = keras.layers.Dense(len(z), weights=[alpha*y, beta*z])
    return [layer(x)]

  @classmethod
  def handle_global_average_pool(cls, node, input_dict):
    x = input_dict[node.inputs[0]]
    dim = K.ndim(x) - 2
    data_format = "channels_first"
    if dim == 1:
      pool = keras.layers.GlobalAveragePooling1D
    elif dim == 2:
      pool = keras.layers.GlobalAveragePooling2D
    elif dim == 3:
      pool = keras.layers.GlobalAveragePooling3D
    else:
      raise NotImplementedError('global avgpooling with dim {} is not implemented.'.format(dim))
    res = pool(data_format=data_format)(x)
    shape = list(K.int_shape(res))[1:] + [1] * dim
    res = keras.layers.Reshape(shape)(res)
    return [res]

  @classmethod
  def handle_global_max_pool(cls, node, input_dict):
    x = input_dict[node.inputs[0]]
    dim = K.ndim(x) - 2
    data_format = "channels_first"
    if dim == 1:
      pool = keras.layers.GlobalMaxPooling1D
    elif dim == 2:
      pool = keras.layers.GlobalMaxPooling2D
    elif dim == 3:
      pool = keras.layers.GlobalMaxPooling3D
    else:
      raise NotImplementedError('global maxpooling with dim {} is not implemented.'.format(dim))
    res = pool(data_format=data_format)(x)
    shape = list(K.int_shape(res))[1:] + [1] * dim
    res = keras.layers.Reshape(shape)(res)
    return [res]

  @classmethod
  def handle_leaky_relu(cls, node, input_dict):
    x = input_dict[node.inputs[0]]
    alpha = node.attrs.get("alpha", 1.0)
    return [Lambda(lambda a: K.relu(a)-alpha*K.relu(-a))(x)]

  @classmethod
  def handle_max(cls, node, input_dict):
    values = [input_dict[a] for a in node.inputs]
    return [Lambda(lambda x: K.max(K.stack(x), axis=0))(values)]

  @classmethod
  def handle_max_pool(cls, node, input_dict):
    x = input_dict[node.inputs[0]]
    dim = K.ndim(x) - 2
    if dim == 1:
        pool = keras.layers.MaxPooling1D
    elif dim == 2:
        pool = keras.layers.MaxPooling2D
    elif dim == 3:
        pool = keras.layers.MaxPooling3D
    else:
        raise NotImplementedError('max pooling with dim {} is not implemented.'.format(dim))
    return cls._pool(node, input_dict, pool)

  @classmethod
  def handle_mean(cls, node, input_dict):
    values = [input_dict[a] for a in node.inputs]
    return [Lambda(lambda x: K.mean(K.stack(x), axis=0))(values)]

  @classmethod
  def handle_min(cls, node, input_dict):
    values = [input_dict[a] for a in node.inputs]
    return [Lambda(lambda x: K.min(K.stack(x), axis=0))(values)]

  @classmethod
  def handle_mul(cls, node, input_dict):
    return [cls._bin_op(node, input_dict, keras.layers.multiply)]

  @classmethod
  def handle_p_relu(cls, node, input_dict):
    x = input_dict[node.inputs[0]]
    slope = input_dict[node.inputs[1]]
    slope = cls._explicit_broadcast(slope, 1, len(x.get_shape()))
    return [keras.layers.advanced_activations.PReLU(alpha=slope)]


  @classmethod
  def handle_pad(cls, node, input_dict):
    num_dim = int(len(node.attrs["pads"])/2)
    mode = node.attrs.get("mode", "constant")
    if mode != "constant":
        warnings.warn("Unsupported mode:{} attribute by Keras in "
                      "pad operator. The attribute will be ignored.".format(mode),
                      UserWarning)

    value = node.attrs.get("value", 0)
    if value != 0:
        warnings.warn("Unsupported value:{} attribute by Keras in "
                      "pad operator. The attribute will be ignored.".format(value),
                      UserWarning)

    pads = node.attrs["pads"]

    x = input_dict[node.inputs[0]]

    return [cls.get_keras_pad(x, pads, num_dim)]

  @classmethod
  def handle_reciprocal(cls, node, input_dict):
    return [Lambda(lambda x: 1.0/x)(input_dict[node.inputs[0]])]

  @classmethod
  def handle_reduce_l1(cls, node, input_dict):
    axis = node.attrs.get("axes")
    if isinstance(axis, list) and len(axis) > 1:
        raise NotImplementedError('reduce_l1 with axis length>1 is not implemented')
    keepdims = node.attrs.get("keepdims", 1)
    if keepdims == 1:
      keepdims = True
    else:
      keepdims = False
    tensor = input_dict[node.inputs[0]]
    return [Lambda(lambda x: K.sum(K.abs(x), axis=axis, keepdims=keepdims))(tensor)]

  @classmethod
  def handle_reshape(cls, node, input_dict):
    tensor = input_dict[node.inputs[0]]
    shape =node.attrs["shape"]
    return [keras.layers.core.Reshape(shape)(tensor)]

  @classmethod
  def handle_rnn(cls, node, input_dict):
    hidden_size = node.attrs["hidden_size"]
    direction = node.attrs['direction']
    go_backwards = False
    if direction == 'bidirectional':
      raise NotImplementedError('bidirectional rnn is not implemented')
    elif direction == 'forward':
      go_backwards = False
    elif direction == 'reverse':
      go_backwards = True
    activation = node.attrs['activations']
    if isinstance(activation, list):
      activation = activation[0]
    output_sequence = node.attrs.get('output_sequence', 0)
    if output_sequence == 0:
      return_sequences = False
    else:
      return_sequences = True
    x = input_dict[node.inputs[0]]
    # num_direction=1
    W = input_dict[node.inputs[1]][0]
    R = input_dict[node.inputs[2]][0]
    if len(node.inputs) == 4:
      B = input_dict[node.inputs[3]][0]
      rnn = keras.layers.recurrent.SimpleRNN(units=hidden_size, activation=activation, use_bias=True,
                                             go_backwards=go_backwards, return_sequences=return_sequences,
                                             weights=[W,R,B])(x)
    elif len(node.inputs) == 3:
      rnn = keras.layers.recurrent.SimpleRNN(units=hidden_size, activation=activation, use_bias=False,
                                             go_backwards=go_backwards, return_sequences=return_sequences,
                                             weights=[W, R])(x)
    else:
      raise NotImplementedError

    if return_sequences:
      # onnx[seq_length, num_directions, batch_size, hidden_size] keras:[samples,timesteps,output_dim]
      res = Lambda(lambda _x: K.expand_dims(K.permute_dimensions(_x, [1,0,2]),axis=1))(rnn)
    else:
      # onnx[num_directions, batch_size, hidden_size] keras[samples,output_dim]
      res = Lambda(lambda _x: K.expand_dims(_x, axis=0))(rnn)
    return [res]

  @classmethod
  def handle_selu(cls, node, input_dict):
    return [keras.layers.Activation(activation='selu')(input_dict[node.inputs[0]])]


  @classmethod
  def handle_shape(cls, node, input_dict):
    return [Lambda(lambda x: K.constant(K.shape(x), dtype='int64'))(input_dict[node.inputs[0]])]


  @classmethod
  def handle_softmax(cls, node, input_dict):
    if 'axis' in node.attrs:
        warnings.warn("Unsupported axis attribute by Keras in "
                      "softmax operator. The attribute will be ignored.",
                      UserWarning)
    #todo axis attr
    return [Lambda(lambda x: K.softmax(x))(input_dict[node.inputs[0]])]

  @classmethod
  def handle_sub(cls, node, input_dict):
    return [cls._bin_op(node, input_dict, keras.layers.subtract)]

  @classmethod
  def handle_sum(cls, node, input_dict):
    values = [input_dict[a] for a in node.inputs]
    return [Lambda(lambda x: K.sum(K.stack(x), axis=0))(values)]

  @classmethod
  def handle_mat_mul(cls, node, input_dict):
    return [Lambda(lambda x, y: K.dot(x,y))(input_dict[node.inputs[0]],
                      input_dict[node.inputs[1]])]


prepare = KerasBackend.prepare

run_node = KerasBackend.run_node

run_model = KerasBackend.run_model

supports_device = KerasBackend.supports_device


