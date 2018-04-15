from onnx import TensorProto

STR_TO_ONNX_TYPE = {
    "float32": TensorProto.FLOAT,
    "uint8": TensorProto.UINT8,
    "int8": TensorProto.INT8,
    "uint16": TensorProto.UINT16,
    "int16": TensorProto.INT16,
    "int32": TensorProto.INT32,
    "int64": TensorProto.INT64,
    "bool": TensorProto.BOOL,
    "float16": TensorProto.FLOAT16,
    "float64": TensorProto.DOUBLE,
    "complex64": TensorProto.COMPLEX64,
    "complex128": TensorProto.COMPLEX128,
}


def convert_shape(shape):
    res = []
    for s in shape:
        if s:
            res.append(s)
        else:
            res.append(-1)
    return res


rename_operator = {
    "Dense": "FC",
    "Conv1D": "Conv",
    "Conv2D": "Conv",
    "Conv3D": "Conv",
    "Conv2DTranspose": "ConvTranspose",
    "Cropping1D": "Crop",
    "Cropping2D": "Crop",
    "Cropping3D": "Crop",
    "UnSampling1D": "Unsample",
    "UnSampling2D": "Unsample",
    "UnSampling3D": "Unsample",
    "ZeroPadding1D": "Pad",
    "ZeroPadding2D": "Pad",
    "ZeroPadding3D": "Pad",
    "MaxPooling1D": "MaxPool",
    "MaxPooling2D": "MaxPool",
    "MaxPooling3D": "MaxPool",
    "AveragePooling1D": "AveratePool",
    "AveragePooling2D": "AveratePool",
    "AveragePooling3D": "AveratePool",
    "GlobalMaxingPooling1D": "GlobalMaxPool",
    "GlobalMaxingPooling2D": "GlobalMaxPool",
    "GlobalAveragePooling1D": "GlobalAveragePool",
    "GlobalAveragePooling2D": "GlobalAveragePool",
    "Subtract": "Sub",
    "Multiply": "Mul",
    "Maximum": "Max",
    "Concatenate": "Concat",
    "softmax": "Softmax",
    "selu": "Selu",
    "softplus": "Softplus",
    "softsign": "Softsign",
    "relu": "Relu",
    "tanh": "Tanh",
    "sigmoid": "Sigmoid",
    "hard_sigmoid": "HardSigmoid",
}
