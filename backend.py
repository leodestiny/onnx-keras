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
import tensorflow as tf
from tensorflow.python.client import device_lib

# TODO: allow more flexible placement
def get_device_option(device):
  m = {DeviceType.CPU: '/cpu',
       DeviceType.CUDA: '/gpu'}
  return m[device.type]

# TODO: Move this into ONNX main library
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

  def caffe2(self, kmap=lambda x: x):
    for k, v in self.items():
      yield caffe2.python.utils.MakeArgument(kmap(k), v)

# TODO: Move this into ONNX main library
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
    # print('---------------node intro----------')
    # print(type(node))
    # print(node)
    # print('--------------node end-------------')
    self.name = str(node.name)
    self.op_type = str(node.op_type)
    self.attrs = OnnxAttributes.from_onnx(node.attribute)
    self.consumed_inputs = self.attrs.pop("consumed_inputs", None)
    self.inputs = list(node.input)
    self.outputs = list(node.output)
    self.node_proto = node


class KerasNet(object):
  """
    Placeholder class for a protobuf definition.
  """
  def __init__(self):
    self.op = []
    self.external_input = []
    self.external_output = []
    self.output = []

    self.output_dict = {}


class KerasBackend(Backend):
  """ Tensorflow Backend for ONNX
  """
  # model = Sequential()

  onnx_tf_attribute_map = {
      "scale": "stddev",
      "high": "maxval",
      "low": "minval",
      "axes": "axis",
      # "keepdims": "keep_dims",
      "axis": "dim",
      "to": "dtype",
  }

  # onnx_tf_per_op_attr_map = {}

  onnx_keras_op_map = {
      "abs": K.abs,
      "cast": K.cast,
      "ceil": np.ceil,
      "exp": K.exp,
      "gather": K.gather,
      "hardsigmoid": K.hard_sigmoid,
      "log": K.log,
      "pow": K.pow,
      "random_normal": K.random_normal,
      "random_uniform": K.random_uniform,
      # "reciprocal": tf.reciprocal,
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

  conv_func = {
    1: keras.layers.convolutional.Conv1D,
    2: keras.layers.convolutional.Conv2D,
    3: keras.layers.convolutional.Conv3D,
  }
  zero_padding_func = {
    1: keras.layers.convolutional.ZeroPadding1D,
    2: keras.layers.convolutional.ZeroPadding2D,
    3: keras.layers.convolutional.ZeroPadding3D,
  }

  type_string_to_keras_type = {
      "float": K.floatx(),
  }

  attr_translator = {
      "dtype": lambda cls, x: cls.tensor_type_to_keras_type[x],
      "keepdims": lambda cls, x: bool(x),
      "to": lambda cls, x: cls.type_string_to_keras_type[x],
  }

  @classmethod
  def get_keras_pad(cls, x, pads, data_format=None):
    num_dim = int(len(pads) / 2)
    pads = list(np.transpose(np.array(pads).reshape([2, num_dim]).astype(np.int32)))
    pads = tuple(tuple(i) for i in pads)
    if num_dim == 1:
      return Lambda(lambda _x: K.temporal_padding(_x, pads))(x)
    elif num_dim == 2:
      return Lambda(lambda _x: K.spatial_2d_padding(_x, pads, data_format))(x)
    elif num_dim == 3:
      return Lambda(lambda _x: K.spatial_3d_padding(_x, pads, data_format))(x)
    else:
      raise NotImplementedError("padding with dim {} is not implemented.".format(num_dim))

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
        y = keras.layers.Input(y)
    broadcast = node.attrs.get("broadcast", 1)
    if broadcast == 0:
      warnings.warn("Definition of {} with broadcast disabled is not "
                    "yet supported.".format(node.type), UserWarning)


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
    # creating placeholders for currently unkown inputs
    for value_info in graph_def.input:
      if value_info.name in initialized:
        continue

      shape = list(d.dim_value for d in
                   value_info.type.tensor_type.shape.dim)

      x = Input(shape=shape[1:], name=value_info.name, dtype=
      cls.tensor_type_enum[value_info.type.tensor_type.elem_type])
      # x = tf.placeholder(cls.tensor_type_enum[
      #     value_info.type.tensor_type.elem_type],
      #                    name=value_info.name, shape=shape)
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
      # print(output_ops)
      # print(curr_node_output_map)
      # print(input_dict)
      # print('output_dict', output_dict)
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

    res_model = Model(inputs=inputs,
                      outputs=outputs)
    print(res_model.layers)

    # print(predict_net, original_input_dict, uninitialized)
    # return TensorflowRep(predict_net, original_input_dict, uninitialized)

  @classmethod
  def onnx_initializer_to_input_dict_items(cls,
                                           initializer,
                                           init_net_name='init'):

    def tensor2list(onnx_tensor):
      # Use the onnx.numpy_helper because the data may be raw
      # return onnx.numpy_helper.to_array(onnx_tensor).flatten().tolist()
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
      return method_to_call(node, input_dict)
    else:
      raise NotImplementedError("{} op is not implemented.".format(node.op_type))

  @classmethod
  def handle_trivial(cls, node, input_dict):
    # print(type(node))
    # print(node)
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
    func = cls.onnx_keras_op_map[cls.op_name_to_lower(node.op_type)]
    if len(node.inputs) == 1:
      inputs = input_dict[node.inputs[0]]
    else:
      inputs = [input_dict[name] for name in node.inputs]
    res = Lambda(lambda a: func(a, *attrs))(inputs)
    return [res]

    # return [cls.onnx_keras_op_map[cls.op_name_to_lower(node.op_type)] \
    #   (*inputs, **attrs)]

  @classmethod
  def handle_add(cls, node, input_dict):
    x = input_dict[node.inputs[0]]
    y = input_dict[node.inputs[1]]
    tmp = np.array([0])

    if type(y) == type(tmp):
      total_num_dim = len(x.get_shape())
      y = cls._explicit_broadcast(y,0,total_num_dim-1)
      return [Lambda(lambda a: K.bias_add(a, K.constant(y)))(x)]

    return [cls._bin_op(node, input_dict, keras.layers.add)]  # is layers.add right?

  @classmethod
  def handle_arg_max(cls, node, input_dict):
    data = input_dict[node.inputs[0]]
    axis = node.attrs["axis"]
    keepdims = node.attrs.get("keepdims", 1)
    if keepdims == 1:
      warnings.warn("Definition of ArgMax with keepdims enabled is "
                    "incompatible between onnx and tensorflow.",
                    UserWarning)

    return [Lambda(lambda x: K.argmax(x, axis=axis))(data)]

  @classmethod
  def handle_arg_min(cls, node, input_dict):
    data = input_dict[node.inputs[0]]
    axis = node.attrs["axis"]
    keepdims = node.attrs.get("keepdims", 1)
    if keepdims == 1:
      warnings.warn("Definition of ArgMin with keepdims enabled is "
                    "incompatible between onnx and tensorflow.",
                    UserWarning)
      return [Lambda(lambda x: K.argmin(x, axis=axis))(data)]

  @classmethod
  def _pool(cls, node, input_dict, pool_layer):

    x = input_dict[node.inputs[0]]
    x_rank = len(x.get_shape())

    support_cuda = cls.supports_device("CUDA")
    # storage_format, compute_format = cls.get_data_format(x_rank, support_cuda)

    kernel_shape = node.attrs["kernel_shape"]
    kernel_shape = tuple(kernel_shape)
    strides = node.attrs["strides"]

    data_format = "channels_first"
    if "pads" in node.attrs.keys():
        x = cls.get_keras_pad(x, node.attrs["pads"], data_format)

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
    total_num_dim = len(x.get_shape())
    scale = cls._explicit_broadcast(input_dict[node.inputs[1]], 1, total_num_dim)
    bias = cls._explicit_broadcast(input_dict[node.inputs[2]], 1, total_num_dim)
    mean = cls._explicit_broadcast(input_dict[node.inputs[3]], 1, total_num_dim)
    variance = cls._explicit_broadcast(input_dict[node.inputs[4]], 1, total_num_dim)
    keras.layers.BatchNormalization
    variance_epsilon = node.attrs.get("epsilon", 0.00001)
    if node.attrs.get("is_test", 0):
      return [tf.nn.batch_normalization(x, mean, variance, bias, scale,
                                        variance_epsilon)]
    if "momentum" in node.attrs.keys():
      warnings.warn("Unsupported momentum attribute by Tensorflow in "
                    "batch_normalization. This attribute will be ignored.",
                    UserWarning)
    if "spatial" in node.attrs.keys():
      warnings.warn("Unsupported spatial attribute by Tensorflow in "
                    "batch_normalization. This attribute will be ignored.",
                    UserWarning)
    # TODO: need to conform to the documentation here
    return [tf.nn.batch_normalization(x, mean, variance, bias, scale,
                                      variance_epsilon)]
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
  def get_perm_from_formats(cls, _from, _to):
    return list(map(lambda x: _from.find(x), _to))

  @classmethod
  def _conv(cls, node, input_dict, transpose=False):
    x = input_dict[node.inputs[0]]

    x_rank = len(x.get_shape())
    print('_conv', x_rank, x.get_shape())
    dim = x_rank - 2
    if dim > 4 or dim < 1:
      raise NotImplementedError('conv of dim {} is not implemented.'.format((dim)))
    data_format = 'channels_first'
    # support_cuda = cls.supports_device("CUDA")
    # print('support_cuda,', support_cuda)
    # data_format = cls.get_data_format(support_cuda)

    W_weights = input_dict[node.inputs[1]]

    filters = len(W_weights)
    W_weights = np.transpose(W_weights, [2, 3, 1, 0])
    dilations = node.attrs.get("dilations", None)
    strides = node.attrs.get("strides", None)
    kernel_size = node.attrs.get("kernel_shape")
    if "group" in node.attrs and node.attrs['group'] != 1:
      raise NotImplementedError("'group' attribute in conv is not implemented")

    if "pads" in node.attrs.keys():
      x = cls.get_keras_pad(x, node.attrs["pads"], data_format)

    if len(node.inputs) == 2:
      convolved = keras.layers.convolutional._Conv(
        rank=dim, filters=filters, kernel_size=kernel_size, data_format=data_format,
        dilation_rate=dilations, strides=strides, use_bias=False, weights=[W_weights])(x)
    else:
      bias = input_dict[node.inputs[2]]
      convolved = keras.layers.convolutional._Conv(
        rank=dim, filters=filters, kernel_size=kernel_size, data_format=data_format,
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

    return [Lambda(lambda a: K.elu(a, alpha))(x)]

  @classmethod
  def handle_equal(cls, node, input_dict):
    return [cls._bin_op(node, input_dict, Lambda(lambda x, y: K.equal(x, y)), inputlist=False)]

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
    shape = K.int_shape(tensor)
    split0, split1 = np.split(shape, [axis, np.size(shape) - axis])
    split0 = np.prod(split0)
    split1 = np.prod(split1)
    output_shape = np.stack([split0, split1])
    return [keras.layers.core.Reshape(output_shape)(tensor)]

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
      raise NotImplementedError('max pooling with dim {} is not implemented.'.format(dim))
    return [pool(data_format=data_format)(x)]

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
      raise NotImplementedError('max pooling with dim {} is not implemented.'.format(dim))
    return [pool(data_format=data_format)(x)]

  @classmethod
  def handle_leaky_relu(cls, node, input_dict):
    x = input_dict[node.inputs[0]]
    if not "alpha" in node.attrs.keys():
      warnings.warn("Provide an alpha value.", UserWarning)
      alpha = 1.0
    else:
      alpha = node.attrs["alpha"]

    return [keras.layers.advanced_activations.LeakyReLU(alpha)(x)]


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

    return [cls.get_keras_pad(x, pads)]

  @classmethod
  def handle_random_normal_like(cls, node, input_dict):
    shape = K.int_shape(input_dict[node.inputs[0]])
    mean = node.attrs["mean"]
    stddev = node.attrs["scale"]
    dtype = cls.tensor_type_to_keras_type[node.attrs["dtype"]]
    seed = node.attrs["seed"] if "seed" in node.attrs.keys() else None
    return [K.random_normal(shape, mean, stddev, dtype, seed)]

  @classmethod
  def handle_random_uniform_like(cls, node, input_dict):
    shape = K.int_shape(input_dict[node.inputs[0]])
    minval = node.attrs["low"]
    maxval = node.attrs["high"]
    dtype = cls.tensor_type_to_keras_type[node.attrs["dtype"]]
    seed = node.attrs["seed"] if "seed" in node.attrs.keys() else None
    return [K.random_uniform(shape, minval, maxval, dtype, seed)]

  @classmethod
  def handle_reduce_l1(cls, node, input_dict):
    axis = node.attrs.get("axes")
    keepdims = node.attrs.get("keepdims", 1)
    tensor = input_dict[node.inputs[0]]
    return [Lambda(lambda x: K.sum(x, axis=axis, keepdims=keepdims))(tensor)]

  @classmethod
  def handle_reduce_l2(cls, node, input_dict):
    axis = node.attrs.get("axes")
    keepdims = node.attrs.get("keepdims", 1)
    tensor = input_dict[node.inputs[0]]
    return [Lambda(lambda x: K.std(x, axis=axis, keepdims=keepdims))(tensor)]

  @classmethod
  def handle_reshape(cls, node, input_dict):
    tensor = input_dict[node.inputs[0]]
    shape =node.attrs["shape"]
    return [keras.layers.core.Reshape(shape)(tensor)]

  @classmethod
  def handle_rnn(cls, ):
    return None

  @classmethod
  def handle_selu(cls, node, input_dict):
    # warnings.warn("Definition of Selu is different "
    #               "between onnx and tensorflow.", UserWarning)
    return [keras.layers.Activation(activation='selu')(input_dict[node.inputs[0]])]


  @classmethod
  def handle_shape(cls, node, input_dict):
    return [Lambda(lambda x: K.constant(K.shape(x), dtype='int64'))(input_dict[node.inputs[0]])]


  @classmethod
  def handle_softmax(cls, node, input_dict):
    # if "axis" in node.attrs:
    #   axis = node.attrs["axis"]
    #   axis = (axis if axis > 0
    #           else len(input_dict[node.inputs[0]].get_shape()) + axis)
    # else:
    #   axis = 1
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

  @classmethod
  def supports_device(cls, device):
    if device == "CUDA":
      local_device_protos = device_lib.list_local_devices()
      return len([x.name for x in
                  local_device_protos if x.device_type == 'GPU']) > 0
    elif device == "CPU":
      return True
    return False

prepare = KerasBackend.prepare

run_node = KerasBackend.run_node

run_model = KerasBackend.run_model

supports_device = KerasBackend.supports_device


