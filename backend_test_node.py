from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import numpy as np
from backend import run_node
from onnx import helper
# from onnx.onnx_pb2 import TensorProto
import skimage.measure as sm

class TestNode(unittest.TestCase):
  """ Tests for nodes
  """

  def _get_rnd(self, shape, low=-1.0, high=1.0):
    return np.random.uniform(low, high, np.prod(shape)) \
                      .reshape(shape) \
                      .astype(np.float32)

  def _elu(self, x):
    # f(x) = alpha * (exp(x) - 1.) for x < 0,
    # f(x) = x for x >= 0
    if x < 0.:
      return np.expm1(x)
    return x

  def _leaky_relu(self, x, alpha):
    # f(x) = alpha * x for x < 0,
    # f(x) = x for x >= 0
    if x < 0.:
      return alpha * x
    return x

  def _logsoftmax_2d(self, x):
    max_x = np.max(x, axis=1).reshape((-1, 1))
    exp_x = np.exp(x - max_x)
    return x - max_x - np.log(np.sum(exp_x, axis=1).reshape((-1, 1)))

  def _softmax_2d(self, x):
    max_x = np.max(x, axis=1).reshape((-1, 1))
    exp_x = np.exp(x - max_x)
    return exp_x / np.sum(exp_x, axis=1).reshape((-1, 1))

  def test_abs(self):
    node_def = helper.make_node("Abs", ["X"], ["Y"])
    x = self._get_rnd([10, 10])
    output = run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], np.abs(x))

  def test_add(self):
    node_def = helper.make_node("Add", ["X", "Y"], ["Z"], broadcast=1, axis=1)
    x = self._get_rnd([20, 3, 5, 5])
    y = self._get_rnd([3])
    output = run_node(node_def, [x, y])

    np.testing.assert_almost_equal(output["Z"],
                                   np.add(x, y.reshape((1,3,1,1))))

  def test_arg_max(self):
    for axis in [0, 1]:
      node_def = helper.make_node(
          "ArgMax", ["data"], ["reduced"], axis=axis, keepdims=0)
      data = self._get_rnd([10, 10])
      output = run_node(node_def, [data])
      np.testing.assert_almost_equal(output["reduced"],
                                     np.argmax(data, axis=axis))

  def test_arg_min(self):
    for axis in [0, 1]:
      node_def = helper.make_node(
          "ArgMin", ["data"], ["reduced"], axis=axis, keepdims=0)
      data = self._get_rnd([10, 10])
      output = run_node(node_def, [data])
      np.testing.assert_almost_equal(output["reduced"],
                                     np.argmin(data, axis=axis))

  def test_average_pool(self):
    # TODO: fix this test
    shape = [10, 1, 40, 40]
    node_def = helper.make_node(
        "AveragePool", ["X"], ["Y"],
        kernel_shape=[1, 2],
        pads=[0, 0],
        strides=[1, 2])
    x = self._get_rnd(shape)
    output = run_node(node_def, [x])
    test_output = np.zeros([10,1,40,20])
    for i1 in range(0, 10):
      for i2 in range(0, 1):
        for j1 in range(0, 40):
          for j2 in range(0, 20):
            test_output[i1][i2][j1][j2] = \
              (x[i1][i2][j1][2*j2] + x[i1][i2][j1][2*j2 + 1]) / 2
    np.testing.assert_almost_equal(output["Y"], test_output)

  def _batch_normalization(self, x, mean, variance, bias, scale,
                           variance_epsilon):
    inv = np.reciprocal(np.sqrt(variance + variance_epsilon))
    if scale is not None:
      inv *= scale
    return x * inv + (bias - mean * inv if bias is not None else -mean * inv)

  def test_batch_normalization(self):
    return
    node_def = helper.make_node(
        "BatchNormalization", ["X", "scale", "bias", "mean", "var"], ["Y"],
        epsilon=0.001)
    x_shape = [3, 5, 4, 2]
    momentum = 0.9
    param_shape = [5]
    _param_shape = [1, 5, 1, 1]
    x = self._get_rnd(x_shape, 0, 1)
    m = self._get_rnd(param_shape, 0, 1)
    _m = m.reshape(_param_shape)
    _m = _m * momentum + np.mean(x, axis=0) * (1 - momentum)
    v = self._get_rnd(param_shape, 0, 1)
    _v = v.reshape(_param_shape)
    _v = _v * momentum + np.var(x, axis=0) * (1 - momentum)
    scale = self._get_rnd(param_shape, 0, 1)
    _scale = scale.reshape(_param_shape)
    bias = self._get_rnd(param_shape, 0, 1)
    _bias = bias.reshape(_param_shape)
    golden = self._batch_normalization(x, _m, _v, _bias, _scale, 0.001)
    output = run_node(node_def, [x, scale, bias, m, v])
    np.testing.assert_almost_equal(output["Y"], golden, decimal=5)

  def test_cast(self):
    return
    # node_def = helper.make_node("Cast", ["input"], ["output"], to=TensorProto.INT8)
    # import onnx
    # print(type(TensorProto.INT8))
    # onnx.checker.check_node(node_def)
    for ty, _type in [(TensorProto.FLOAT,
                         'float32'), (TensorProto.UINT8,
                                       'uint8'), (TensorProto.INT8, 'int8'),
                        (TensorProto.UINT16,
                         'uint16'), (TensorProto.INT16,
                                      'int16'), (TensorProto.INT32, 'int32'),
                        (TensorProto.INT64,
                         'int64'), (TensorProto.BOOL,
                                     'bool'), (TensorProto.FLOAT16,
                                                'float16'),
                        (TensorProto.DOUBLE,
                         'float64'), (TensorProto.COMPLEX64,
                                       'complex64'), (TensorProto.COMPLEX128,
                                                       'complex128')]:
      node_def = helper.make_node("Cast", ["input"], ["output"], to=ty)
      vector = [2, 3]
      output = run_node(node_def, [vector])
      np.testing.assert_equal(output["output"].dtype, _type)

  def test_clip(self):
      node_def = helper.make_node(
          'Clip',
          inputs=['X'],
          outputs=['Y'],
          min=-1.0,
          max=1.0
      )
      x = np.random.randn(3, 4, 5).astype(np.float32)
      output = run_node(node_def, [x])
      np.testing.assert_almost_equal(output["Y"], np.clip(x, -1.0, 1.0))

  def test_concat(self):
    shape = [10, 20, 5]
    for axis in range(1, len(shape)):
      node_def = helper.make_node("Concat", ["X1", "X2"], ["Y"], axis=axis)
      x1 = self._get_rnd(shape)
      x2 = self._get_rnd(shape)
      output = run_node(node_def, [x1, x2], [0, 1])
      np.testing.assert_almost_equal(output["Y"], np.concatenate((x1, x2),
                                                                axis))

  def test_conv(self):
    # return
    N, C, H, W = 4, 3, 5, 5
    x_shape = [N, C, H, W]
    K, kH, kW = 6, 3, 3
    weight_shape = [K, C, kH, kW]
    node_def = helper.make_node(
        "Conv", ["X", "weights"], ["Y"],
        pads=[1, 1, 1, 1],
        kernel_shape=[kH, kW])

    x = self._get_rnd(x_shape)
    weights = self._get_rnd(weight_shape)
    output = run_node(node_def, [x, weights])

    out_shape = [N, K, H, W]
    test_output = np.zeros(out_shape)
    for n in range(N):
      for c in range(C):
        for h in range(H):
          for w in range(W):
            for k in range(K):
              for kh in range(kH):
                for kw in range(kW):
                  h_in_range = (h - kH // 2 + kh) < H and (
                      h - kH // 2 + kh) >= 0
                  w_in_range = (w - kW // 2 + kw) < W and (
                      w - kW // 2 + kw) >= 0
                  if h_in_range and w_in_range:
                    test_output[n][k][h][w] += (x[n][c][h - kH // 2 + kh][
                        w - kW // 2 + kw] * weights[k][c][kh][kw])

    np.testing.assert_almost_equal(output["Y"], test_output, decimal=5)

  def test_depth_to_space(self):
    b,c,h,w = shape = (2,8,3,3)
    blocksize = 2
    node_def = helper.make_node("DepthToSpace", ["X"], ["Y"], blocksize=blocksize)
    x = self._get_rnd(shape)
    tmp = np.reshape(x, [b, blocksize, blocksize, c // (blocksize ** 2), h, w])
    tmp = np.transpose(tmp, [0, 3, 4, 1, 5, 2])
    test_output = np.reshape(tmp, [b, c // (blocksize ** 2), h * blocksize, w * blocksize])
    output = run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], test_output)

  def test_elu(self):
    node_def = helper.make_node("Elu", ["X"], ["Y"])
    x = self._get_rnd([10, 10])
    output = run_node(node_def, [x])
    test_output = [[self._elu(b) for b in a ] for a in x]
    np.testing.assert_almost_equal(output["Y"], test_output)

  def test_equal(self):
    node_def = helper.make_node("Equal", ["X", "Y"], ["Z"], broadcast=1, axis=1)
    x = self._get_rnd([5, 3, 3, 2])
    y = self._get_rnd([3, 3])
    output = run_node(node_def, [x, x], [0, 1])
    np.testing.assert_equal(output["Z"], np.equal(x, x))
    # output = run_node(node_def, [x, y], [0, 1])
    # np.testing.assert_equal(output["Z"], np.equal(x, np.reshape(
    #     y, [1, 3, 3, 1])))

  def test_exp(self):
    node_def = helper.make_node("Exp", ["X"], ["Y"])
    x = self._get_rnd([10, 10])
    x = x - 3.6
    output = run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], np.exp(x))

  def test_flatten(self):
    # If input tensor has shape (d_0, d_1, ... d_n) then the
    # output will have shape:
    #
    # (d_0 X d_1 ... d_(axis-1), d_axis X d_(axis+1) ... X dn)
    #
    # TODO: pass axis attribute which is supported in newer
    # versions of onnx
    node_def = helper.make_node("Flatten", ["X"], ["Y"])
    x = self._get_rnd([10, 2, 3, 4, 5])
    output = run_node(node_def, [x])
    # TODO: pass axis=3 and uncomment the line below
    # np.testing.assert_almost_equal(output["Y"], x.reshape([60, 20]))
    np.testing.assert_almost_equal(output["Y"], x.reshape([10, 120]))
    # node_def = helper.make_node("Flatten", ["X"], ["Y"], axis=3)
    # output = run_node(node_def, [x])
    # np.testing.assert_almost_equal(output["Y"], x.reshape([60, 20]))

  def test_gather(self):
    return
    node_def = helper.make_node("Gather", ["X", "Y"], ["Z"])
    x = self._get_rnd([10, 10])
    y = np.array([[0, 1], [1, 2]])
    output = run_node(node_def, [x, y], [0, 1])
    test_output = np.zeros((2, 2, 10))
    for i in range(0, 2):
      for j in range(0, 10):
        test_output[0][i][j] = x[i][j]
    for i in range(0, 2):
      for j in range(0, 10):
        test_output[1][i][j] = x[i + 1][j]
    np.testing.assert_almost_equal(output["Z"], test_output)

  def test_gemm(self):
      node_def = helper.make_node(
          "Gemm", ["A", "B", "C"], ["Y"],
          transA=0,
          transB=0,
          broadcast=1,
          alpha=1.0,
          beta=1.0)
      x = np.floor(self._get_rnd([20, 10]))
      y = np.floor(self._get_rnd([10, 10]))
      z = np.floor(self._get_rnd([10]))
      output = run_node(node_def, [x, y, z])
      test_output = np.matmul(x, y) + z
      np.testing.assert_almost_equal(output["Y"], test_output)

  def test_global_average_pool(self):
    #   Image case:  (N x C x H x W), where N is the batch size,
    # C is the number of channels, and H and W are the height
    # and the width of the data
    #
    #   Non-image case: (N x C x D1 x D2 ... Dn)
    #
    #   Output data tensor from pooling across the input tensor.
    # Dimensions will be N x C x 1 x 1
    node_def = helper.make_node("GlobalAveragePool", ["X"], ["Y"])
    x = self._get_rnd([10, 10, 2, 3])
    output = run_node(node_def, [x])
    test_output = np.zeros([10, 10, 1, 1])
    for i1 in range(0, 10):
      for i2 in range(0, 10):
        sum = 0
        for j1 in range(0, 2):
          for j2 in range(0, 3):
            sum += x[i1][i2][j1][j2]
        test_output[i1][i2][0][0] = sum / 6.
    np.testing.assert_almost_equal(output["Y"], test_output)

  def test_global_max_pool(self):
    #   Image case:  (N x C x H x W), where N is the batch size,
    # C is the number of channels, and H and W are the height
    # and the width of the data
    #
    #   Non-image case: (N x C x D1 x D2 ... Dn)
    #
    #   Output data tensor from pooling across the input tensor.
    # Dimensions will be N x C x 1 x 1
    node_def = helper.make_node("GlobalMaxPool", ["X"], ["Y"])
    x = self._get_rnd([10, 10, 2, 3])
    output = run_node(node_def, [x])
    test_output = np.zeros([10, 10, 1, 1])
    for i1 in range(0, 10):
      for i2 in range(0, 10):
        max = x[i1][i2][0][0]
        for j1 in range(0, 2):
          for j2 in range(0, 3):
            if max < x[i1][i2][j1][j2]:
              max = x[i1][i2][j1][j2]
        test_output[i1][i2][0][0] = max
    np.testing.assert_almost_equal(output["Y"], test_output)

  def test_hard_sigmoid(self):
    node_def = helper.make_node("HardSigmoid", ["X"], ["Y"], alpha=0.5, beta=0.6)
    x = self._get_rnd((10,5,3))
    test_output = np.clip(x*0.5+0.6, 0, 1)
    output = run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], test_output)

  def test_leakyrelu(self):
    node_def = helper.make_node("LeakyRelu", ["X"], ["Y"], alpha=2.0)
    x = self._get_rnd([10, 10])
    output = run_node(node_def, [x])
    test_output = [[self._leaky_relu(b, 2.0) for b in a] for a in x]
    np.testing.assert_almost_equal(output["Y"], test_output)

  def test_log(self):
    node_def = helper.make_node("Log", ["X"], ["Y"])
    x = self._get_rnd([10,10])
    x = x + 3.6
    output = run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], np.log(x))

  def test_log_softmax(self):
    node_def = helper.make_node("LogSoftmax", ["X"], ["Y"], axis=1)
    x = np.abs(self._get_rnd((3,4,5)))
    test_output = self._logsoftmax_2d(x.reshape(3,20)).reshape(3,4,5)
    output = run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], test_output, decimal=6)

  def test_max(self):
    node_def = helper.make_node("Max", ["X1", "X2", "X3", "X4"], ["Z"])
    x1 = self._get_rnd([10, 10])
    x2 = self._get_rnd([10, 10])
    x3 = self._get_rnd([10, 10])
    x4 = self._get_rnd([10, 10])
    output = run_node(node_def, [x1, x2, x3, x4], [0,1,2,3])
    test_output = np.maximum(np.maximum(np.maximum(x1, x2), x3), x4)
    np.testing.assert_almost_equal(output["Z"], test_output)

  def test_mat_mul(self):
    return
    node_def = helper.make_node("MatMul", ["X1", "X2"], ["Z"])
    x1 = self._get_rnd([10, 20])
    x2 = self._get_rnd([20, 15])
    test_output = np.matmul(x1,x2)
    print(test_output.shape)
    output = run_node(node_def, [x1,x2], [0,1])
    np.testing.assert_almost_equal(output["Z"], test_output)

  def test_max_pool(self):
    node_def = helper.make_node(
        "MaxPool", ["X"], ["Y"],
        kernel_shape=[1, 2],
        pads=[0, 0],
        strides=[1, 2])
    x = self._get_rnd([10, 10, 4, 4])
    output = run_node(node_def, [x])
    test_output = np.zeros([10, 10, 4, 2])
    for i1 in range(0, 10):
      for i2 in range(0, 10):
        for j1 in range(0, 4):
          for j2 in range(0, 2):
            test_output[i1][i2][j1][j2] = \
              max(x[i1][i2][j1][2*j2], x[i1][i2][j1][2*j2 + 1])
    np.testing.assert_almost_equal(output["Y"], test_output)

  def test_min(self):
    node_def = helper.make_node("Min", ["X1", "X2", "X3", "X4"], ["Z"])
    x1 = self._get_rnd([10, 10])
    x2 = self._get_rnd([10, 10])
    x3 = self._get_rnd([10, 10])
    x4 = self._get_rnd([10, 10])
    output = run_node(node_def, [x1, x2, x3, x4], [0,1,2,3])
    test_output = np.minimum(np.minimum(np.minimum(x1, x2), x3), x4)
    np.testing.assert_almost_equal(output["Z"], test_output)

  def test_mul(self):
    node_def = helper.make_node("Mul", ["X", "Y"], ["Z"], broadcast=1, axis=1)
    x = self._get_rnd([20, 10, 5, 6])
    y = self._get_rnd([20, 10])
    output = run_node(node_def, [x, y], [0, 1])
    np.testing.assert_almost_equal(output["Z"],
                                   np.multiply(x, y.reshape([20, 10, 1, 1])))

  def test_pad(self):
    return
    node_def = helper.make_node(
        "Pad", ["X"], ["Y"], mode="constant", pads=[1, 1, 1, 1], value=0.0)
    x = self._get_rnd([100, 100])
    output = run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"],
                                   np.lib.pad(
                                       x, ((1, 1), (1, 1)),
                                       'constant',
                                       constant_values=(0, 0)))

  def test_reciprocal(self):
    node_def = helper.make_node("Reciprocal", ["X"], ["Y"])
    x = self._get_rnd([10,10])
    output = run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], 1.0 / x)

  def test_reduce_l1(self):
    node_def = helper.make_node("ReduceL1", ["X"], ["Y"], axes=[2])
    x = self._get_rnd([1, 10, 10, 3])
    output = run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"],
                                   np.linalg.norm(x, 1, 2, True))

  def test_reduce_log_sum(self):
    node_def = helper.make_node("ReduceLogSum", ["X"], ["Y"], axes=[1, 2])
    x = np.abs(self._get_rnd([5, 10, 10, 3]))
    output = run_node(node_def, [x])
    np.testing.assert_allclose(
        output["Y"],
        np.log(np.sum(x, axis=(1, 2), keepdims=True)),
        rtol=1e-3)

  def test_reduce_log_sum_exp(self):
    node_def = helper.make_node("ReduceLogSumExp", ["X"], ["Y"], axes=[1, 2])
    x = self._get_rnd([5, 10, 10, 3])
    output = run_node(node_def, [x])
    np.testing.assert_allclose(
        output["Y"],
        np.log(np.sum(np.exp(x), axis=(1, 2), keepdims=True)),
        rtol=1e-3)

  def test_reduce_max(self):
    node_def = helper.make_node("ReduceMax", ["X"], ["Y"], axes=[1,3])
    x = self._get_rnd([5, 10, 10, 3])
    output = run_node(node_def, [x])
    np.testing.assert_allclose(
        output["Y"], np.max(x, (1, 3), keepdims=True), rtol=1e-3)

  def test_reduce_mean(self):
    node_def = helper.make_node("ReduceMean", ["X"], ["Y"], axes=[1, 2])
    x = self._get_rnd([5, 10, 10, 3])
    output = run_node(node_def, [x])
    np.testing.assert_allclose(
        output["Y"], np.mean(x, (1, 2), keepdims=True), rtol=1e-3)

  def test_reduce_min(self):
    node_def = helper.make_node("ReduceMin", ["X"], ["Y"], axes=[1, 2])
    x = self._get_rnd([5, 10, 10, 3])
    output = run_node(node_def, [x])
    np.testing.assert_allclose(
        output["Y"], np.min(x, (1, 2), keepdims=True), rtol=1e-3)

  def test_reduce_prod(self):
    node_def = helper.make_node("ReduceProd", ["X"], ["Y"], axes=[1, 2])
    x = self._get_rnd([1, 5, 5, 3])
    output = run_node(node_def, [x])
    np.testing.assert_allclose(
        output["Y"], np.prod(x, (1, 2), keepdims=True), rtol=1e-3)

  def test_reduce_sum(self):
    node_def = helper.make_node("ReduceSum", ["X"], ["Y"], axes=[1,2])
    x = self._get_rnd([5, 10, 10, 3])
    output = run_node(node_def, [x])
    np.testing.assert_allclose(
        output["Y"], np.sum(x, (1, 2), keepdims=True), rtol=1e-3)

  def test_reduce_sum_square(self):
    node_def = helper.make_node("ReduceSumSquare", ["X"], ["Y"], axes=[1,2])
    x = self._get_rnd([5, 10, 10, 3])
    output = run_node(node_def, [x])
    np.testing.assert_allclose(
        output["Y"], np.sum(np.square(x), (1, 2), keepdims=True), rtol=1e-3)

  def test_relu(self):
    node_def = helper.make_node("Relu", ["X"], ["Y"])
    x = self._get_rnd([10,10])
    output = run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], np.maximum(x, 0))

  def test_reshape(self):
    node_def = helper.make_node("Reshape", ["X"], ["Y"], shape=[10,4,5])
    x = self._get_rnd([10,20])
    output = run_node(node_def, [x], [0])
    np.testing.assert_almost_equal(output["Y"], x.reshape(10,4,5))

  def test_sigmoid(self):
    node_def = helper.make_node("Sigmoid", ["X"], ["Y"])
    x = self._get_rnd([10,10])
    output = run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], 1 / (1 + np.exp(-x)))

  def test_softmax(self):
    node_def = helper.make_node("Softmax", ["X"], ["Y"])
    x = self._get_rnd((3, 20))
    test_output = self._softmax_2d(x)
    output = run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], test_output, decimal=6)

  def test_softplus(self):
    node_def = helper.make_node("Softplus", ["X"], ["Y"])
    x = self._get_rnd([3, 4, 5])
    output = run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], np.log(np.exp(x) + 1))

  def test_softsign(self):
    node_def = helper.make_node("Softsign", ["X"], ["Y"])
    x = self._get_rnd([3, 4, 5])
    output = run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], x / (1 + np.abs(x)))

  def test_space_to_depth(self):
    node_def = helper.make_node("SpaceToDepth", ["X"], ["Y"], blocksize=2)
    x_shape = [1, 3, 10, 10]
    x = self._get_rnd(x_shape)
    output = run_node(node_def, [x])
    y = np.reshape(
      np.swapaxes(x.reshape(1, 3, 2, 5, 2, 5), 3, 4), (1, 12,5,5))
    np.testing.assert_allclose(output["Y"], y, rtol=1e-3)

  def test_squeeze(self):
    node_def = helper.make_node("Squeeze", ["X"], ["Y"], axes=[2,3])
    x = self._get_rnd([10,5,1,1,30])
    # x = np.array([[[0], [1], [2]]])
    output = run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], np.squeeze(x, axis=(2,3)))

  def test_sqrt(self):
    node_def = helper.make_node("Sqrt", ["X"], ["Y"])
    x = self._get_rnd([10,10]) + 1.0
    output = run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], np.sqrt(x), decimal=5)

  def test_sub(self):
      node_def = helper.make_node("Sub", ["X", "Y"], ["Z"], broadcast=1)
      x = self._get_rnd([10, 10])
      y = self._get_rnd([10, 10])
      output = run_node(node_def, [x, y], [0, 1])
      np.testing.assert_almost_equal(output["Z"], np.subtract(x, y))

  def test_sum(self):
      node_def = helper.make_node("Sum", ["X1", "X2", "X3", "X4"], ["Z"])
      x1 = self._get_rnd([10, 10])
      x2 = self._get_rnd([10, 10])
      x3 = self._get_rnd([10, 10])
      x4 = self._get_rnd([10, 10])
      output = run_node(node_def, [x1, x2, x3, x4], [0,1,2,3])
      test_output = x1 + x2 + x3 + x4
      np.testing.assert_almost_equal(output["Z"], test_output)

  def test_tanh(self):
    node_def = helper.make_node("Tanh", ["X"], ["Y"])
    x = self._get_rnd([10,10]) + 1.0
    output = run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], np.tanh(x), decimal=5)

  def test_tile(self):
    return
    node_def = helper.make_node("Tile", ["X1", "X2"], ["Z"])
    x = self._get_rnd([3, 5, 5, 3])
    repeats = [1, 1, 2, 1]
    output = run_node(node_def, [x, repeats], [0, 1])
    np.testing.assert_allclose(output["Z"], np.tile(x, repeats), rtol=1e-3)

  def test_transpose(self):
    node_def = helper.make_node("Transpose", ["X"], ["Y"], perm=[0, 2, 1])
    x = self._get_rnd([1000]).reshape([10, 10, 10])
    output = run_node(node_def, [x])
    np.testing.assert_almost_equal(output["Y"], np.transpose(x, (0, 2, 1)))

if __name__ == '__main__':
  unittest.main()

