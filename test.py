import keras
import onnx
from keras.models import *
from keras.layers import *
from frontend import *
from onnx.helper import *
from onnx.checker import check_model

model = Sequential()
model.add(Conv2D(64,(7,7),strides=(2,2),input_shape=(3,128,128),data_format="channels_first"))

m = keras_model_to_onnx_model(model)

with open("test.onnx","wb") as f:
    f.write(m.SerializeToString())