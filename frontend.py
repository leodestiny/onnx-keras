import numpy as np
from keras.layers import *
from onnx.checker import check_model
from onnx.helper import (make_graph, make_model, make_node, make_tensor, make_tensor_value_info,make_opsetid)
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE

from utils import STR_TO_ONNX_TYPE, convert_shape, rename_operator


class KerasFrontend(object):

    @classmethod
    def make_weights(cls, name, weight):
        return make_tensor(name=name,
                           data_type=NP_TYPE_TO_TENSOR_TYPE[weight.dtype],
                           dims=convert_shape(weight.shape),
                           vals=weight.flatten().tolist())

    @classmethod
    def make_symbolic_weights(cls, name, weights):
        return make_tensor_value_info(name=name,
                                      elem_type=NP_TYPE_TO_TENSOR_TYPE[weights.dtype],
                                      shape=convert_shape(weights.shape))

    @classmethod
    def switch_onnx_node_creater(cls, layer):
        class_to_handler = {
            Dense: cls.create_dense,
            Conv1D: cls.create_conv1D,
            Conv2D: cls.create_conv2D,
            Conv3D: cls.create_conv3D,
            Conv2DTranspose: cls.create_conv2D_transpose,
            Cropping1D: cls.create_cropping1D,
            Cropping2D: cls.create_cropping3D,
            Cropping3D: cls.create_cropping3D,
            UpSampling1D: cls.create_upsampling1D,
            UpSampling2D: cls.create_upsampling2D,
            UpSampling3D: cls.create_upsampling3D,
            ZeroPadding1D: cls.create_zero_padding1D,
            ZeroPadding2D: cls.create_zero_padding2D,
            ZeroPadding3D: cls.create_zero_padding3D,
            MaxPooling1D: cls.create_max_pooling1D,
            MaxPooling2D: cls.create_max_pooling2D,
            MaxPooling3D: cls.create_max_pooling3D,
            AveragePooling1D: cls.create_average_pooling1D,
            AveragePooling2D: cls.create_average_pooling2D,
            AveragePooling3D: cls.create_average_pooling3D,
            GlobalMaxPooling1D: cls.create_global_max_pooling1D,
            GlobalMaxPooling2D: cls.create_global_max_pooling2D,
            GlobalAveragePooling1D: cls.create_global_average_pooling1D,
            GlobalAveragePooling2D: cls.create_global_average_pooling2D,
            Add: cls.create_add,
            Subtract: cls.create_sub,
            Multiply: cls.create_mul,
            Maximum: cls.create_max,
            Concatenate: cls.create_concate,
            LeakyReLU: cls.create_leaky_relu,
            PReLU: cls.create_prelu,
            ELU: cls.create_elu,
            ThresholdedReLU: cls.create_threshoulded_relu,
            Dropout: cls.create_dropout,
            Flatten: cls.create_flatten,
            Reshape: cls.create_reshape,
            BatchNormalization: cls.create_batch_normalization,
            LSTM: cls.create_LSTM,
            SimpleRNN: cls.create_RNN,
            GRU: cls.create_GRU
        }
        activation_to_handler = {
            "softmax": cls.create_softmax,
            "selu": cls.create_selu,
            "softplus": cls.create_softplus,
            "softsign": cls.create_softsign,
            "relu": cls.create_relu,
            "tanh": cls.create_tanh,
            "sigmoid": cls.create_sigmoid,
            "hard_sigmoid": cls.create_hard_sigmoid,
        }

        if layer.__class__ in class_to_handler:
            return class_to_handler[layer.__class__]
        elif layer.__class__ == Activation:
            if layer.get_config()['activation'] in activation_to_handler:
                return activation_to_handler[layer.get_config()['activation']]
            else:
                raise NotImplementedError(
                    "This activation %s is not supported in this version" % (layer.get_config()['activation']))
        else:
            raise NotImplementedError("This layer %s is not supported in this version" % (layer.__class__))

    @classmethod
    def keras_model_to_onnx_model(cls, model,
                                  producer_name="onnx-keras",
                                  model_name="keras-model",
                                  model_version=1):

        # TODO save domain, model_version,doc_string

        opset = make_opsetid("",6)

        model = make_model(cls.keras_graph_to_onnx_graph(model,name=model_name),
                           model_version=model_version,
                           producer_name=producer_name,
                           opset_imports=[opset])
        check_model(model)
        return model

    @classmethod
    def keras_graph_to_onnx_graph(cls, model,name):

        # some import attribute:keras_version, backend

        # this list record outputs of this graph
        graph_outputs = []
        for o in model.outputs:
            graph_outputs.append(make_tensor_value_info(name=o.name,
                                                        elem_type=STR_TO_ONNX_TYPE[K.dtype(o)],
                                                        shape=convert_shape(K.int_shape(o))))

        # this list record input of this graph
        graph_inputs = []
        for i in model.inputs:
            shape = convert_shape(K.int_shape(i))
            # channels_last to channels_first
            if K.image_data_format() == "channels_last" and len(shape) == 4:
                shape = shape[:1] + shape[3:] + shape[1:3]
            graph_inputs.append(make_tensor_value_info(name=i.name,
                                                       elem_type=STR_TO_ONNX_TYPE[K.dtype(i)],
                                                       shape=shape))

        # save structure of the graph (all layers) into nodes
        nodes = []

        # save all weights into initializer
        initializer = []

        for i, layer in enumerate(model.layers):
            # the InputLayer only contain input data of model
            if isinstance(layer, InputLayer):
                continue
            else:
                handler = cls.switch_onnx_node_creater(layer)
                graph_input_list, weight_list, node_list = handler(layer)

                nodes.extend(node_list)
                graph_inputs.extend(graph_input_list)
                initializer.extend(weight_list)

        return make_graph(nodes=nodes,
                          name=name,
                          inputs=graph_inputs,
                          outputs=graph_outputs,
                          initializer=initializer)

    @classmethod
    def create_batch_normalization(cls, layer):
        node_list = []
        graph_input_list = []
        weight_list = []

        config = layer.get_config()

        # onnx attributes
        epsilon = config['epsilon']
        momentum = config['momentum']
        spatial = 1  # default value in onnx

        # weights
        symbolic_weights = layer.weights
        weights_values = K.batch_get_value(symbolic_weights)

        # onnx inputs
        X_name = layer.input.name
        inputs = [X_name]

        if config['center'] and config['scale']:
            # the order of weights is : [gamma(scale),beta(bias), mean, variance]
            scale_name = symbolic_weights[0].name
            B_name = symbolic_weights[1].name
            mean_name = symbolic_weights[2].name
            var_name = symbolic_weights[3].name

            scale_weight, B_weight, mean_weight, var_weight = weights_values

        elif config['center'] and not config['scale']:
            # the order of weights is : [beta(bias), mean, variance]
            B_name = symbolic_weights[0].name
            mean_name = symbolic_weights[1].name
            var_name = symbolic_weights[2].name

            B_weight, mean_weight, var_weight = weights_values

            scale_name = layer.name + "/gamma"
            scale_weight = np.ones(B_weight.shape, dtype=float)

        elif not config['center'] and config['scale']:
            # the order of weights is : [gamma(scale), mean, variance]
            scale_name = symbolic_weights[0].name
            mean_name = symbolic_weights[1].name
            var_name = symbolic_weights[2].name

            scale_weight, mean_weight, var_weight = weights_values

            B_name = layer.name + "/beta"
            B_weight = np.zeros(scale_weight.shape, dtype=float)

        else:
            # the order of weights is : [mean, variance]
            mean_name = symbolic_weights[0].name
            var_name = symbolic_weights[1].name

            mean_weight, var_weight = weights_values

            scale_name = layer.name + "/gamma"
            scale_weight = np.ones(mean_weight.shape, dtype=float)

            B_name = layer.name + "/beta"
            B_weight = np.zeros(mean_weight.shape, dtype=float)

        graph_input_list.append(cls.make_symbolic_weights(scale_name, scale_weight))
        graph_input_list.append(cls.make_symbolic_weights(B_name, B_weight))
        graph_input_list.append(cls.make_symbolic_weights(mean_name, mean_weight))
        graph_input_list.append(cls.make_symbolic_weights(var_name, var_weight))

        weight_list.append(cls.make_weights(scale_name, scale_weight))
        weight_list.append(cls.make_weights(B_name, B_weight))
        weight_list.append(cls.make_weights(mean_name, mean_weight))
        weight_list.append(cls.make_weights(var_name, var_weight))
        inputs.extend([scale_name, B_name, mean_name, var_name])
        consumed_inputs = [0, 0, 0, 1, 1]

        # onnx outpus
        # notate onnx has other opitional output, but there is only one output in keras
        Y_name = layer.output.name
        outputs = [Y_name]

        node = make_node("BatchNormalization",
                         inputs=inputs,
                         outputs=outputs,
                         name=layer.name,
                         epsilon=epsilon,
                         momentum=momentum,
                         consumed_inputs=consumed_inputs,
                         spatial=spatial)
        node_list.append(node)

        return graph_input_list, weight_list, node_list

    @classmethod
    def create_LSTM(cls, layer):
        node_list = []
        graph_input_list = []
        weight_list = []

        config = layer.get_config()

        # onnx attribute

        # no alpha and beta in Keras activation function
        activation_alpha = [0.0, 0.0, 0.0]
        activation_beta = [0.0, 0.0, 0.0]

        # the activation function order is f, g, h in onnx
        # coresponding is recurrent_activation, activation, activation in keras
        activations = [config['recurrent_activation'], config['activation'], config['activation']]

        # no clip in Keras
        clip = None

        if config['go_backwards']:
            direction = "reverse"
        else:
            direction = "forward"

        hidden_size = config['units']

        output_sequence = 0
        if config['return_sequences']:
            output_sequence = 1

        # onnx input
        # however W,R,B matrix order is [i f c o] in keras
        # the W,R,B matrix order is [i o f c] in onnx
        symbolic_weights = layer.weights
        weights_values = K.batch_get_value(symbolic_weights)

        W_name = symbolic_weights[0].name
        W_weight = weights_values[0]
        W_weight = np.concatenate(
            (W_weight[:, :hidden_size], W_weight[:, 3 * hidden_size:], W_weight[:, hidden_size:3 * hidden_size]),
            axis=1)
        graph_input_list.append(cls.make_symbolic_weights(W_name, W_weight))
        weight_list.append(cls.make_weights(W_name, W_weight))

        R_name = symbolic_weights[1].name
        R_weight = weights_values[1]
        R_weight = np.concatenate(
            (R_weight[:, :hidden_size], R_weight[:, 3 * hidden_size:], R_weight[:, hidden_size:3 * hidden_size]),
            axis=1)
        graph_input_list.append(cls.make_symbolic_weights(R_name, R_weight))
        weight_list.append(cls.make_weights(R_name, R_weight))

        X = layer.input.name
        inputs = [X, W_name, R_name]

        if config['use_bias']:
            B_name = symbolic_weights[2].name
            B_weight = weights_values[2]
            B_weight = np.concatenate(
                (B_weight[:hidden_size], B_weight[3 * hidden_size:], B_weight[hidden_size:3 * hidden_size]), axis=0)
            graph_input_list.append(cls.make_symbolic_weights(B_name, B_weight))
            weight_list.append(cls.make_weights(B_name, B_weight))

            inputs.append(B_name)

        # these inputs are not used
        sequence_lens = None
        initial_h = None
        inital_c = None
        P = None

        # onnx output
        # Y represent all intermediate value
        # Y should used when return_sequences is true
        # Y_h represent last output value of hidden
        # set by return_sequence

        Y = ""
        Y_h = ""

        if output_sequence == 1:
            Y = layer.output.name
        else:
            Y_h = layer.output.name
        outputs = [Y, Y_h]

        node_list.append(make_node("LSTM",
                                   inputs=inputs,
                                   outputs=outputs,
                                   name=layer.name,
                                   activations=activations,
                                   activation_alpha=activation_alpha,
                                   activation_beta=activation_beta,
                                   direction=direction,
                                   hidden_size=hidden_size,
                                   output_sequence=output_sequence))

        return graph_input_list, weight_list, node_list

    @classmethod
    def create_GRU(cls, layer):
        node_list = []
        graph_input_list = []
        weight_list = []

        config = layer.get_config()

        # onnx attribute

        # no alpha and beta in Keras activation function
        activation_alpha = [0.0, 0.0]
        activation_beta = [0.0, 0.0]

        # the activation function order is f, g in onnx
        # coresponding is recurrent_activation, activation in keras
        activations = [config['recurrent_activation'], config['activation']]

        # no clip in Keras
        clip = None

        if config['go_backwards']:
            direction = "reverse"
        else:
            direction = "forward"

        hidden_size = config['units']

        output_sequence = 0
        if config['return_sequences']:
            output_sequence = 1

        # onnx input
        # onnx and keras has same order : zrh
        symbolic_weights = layer.weights
        weights_values = K.batch_get_value(symbolic_weights)

        W_name = symbolic_weights[0].name
        W_weight = weights_values[0]
        graph_input_list.append(cls.make_symbolic_weights(W_name, W_weight))
        weight_list.append(cls.make_weights(W_name, W_weight))

        R_name = symbolic_weights[1].name
        R_weight = weights_values[1]
        graph_input_list.append(cls.make_symbolic_weights(R_name, R_weight))
        weight_list.append(cls.make_weights(R_name, R_weight))

        X = layer.input.name
        inputs = [X, W_name, R_name]

        if config['use_bias']:
            B_name = symbolic_weights[2].name
            B_weight = weights_values[2]
            graph_input_list.append(cls.make_symbolic_weights(B_name, B_weight))
            weight_list.append(cls.make_weights(B_name, B_weight))

            inputs.append(B_name)

        # these inputs are not used
        sequence_lens = None

        # onnx output
        # Y represent all intermediate value
        # Y should used when return_sequences is true
        # Y_h represent last output value of hidden
        # set by return_sequence

        Y = ""
        Y_h = ""

        if output_sequence == 1:
            Y = layer.output.name
        else:
            Y_h = layer.output.name
        outputs = [Y, Y_h]

        node_list.append(make_node("GRU",
                                   inputs=inputs,
                                   outputs=outputs,
                                   name=layer.name,
                                   activations=activations,
                                   activation_alpha=activation_alpha,
                                   activation_beta=activation_beta,
                                   direction=direction,
                                   hidden_size=hidden_size,
                                   output_sequence=output_sequence))

        return graph_input_list, weight_list, node_list

    @classmethod
    def create_RNN(cls, layer):
        node_list = []
        graph_input_list = []
        weight_list = []

        config = layer.get_config()

        # onnx attribute

        # no alpha and beta in Keras activation function
        activation_alpha = [0.0]
        activation_beta = [0.0]

        activations = [config['activation']]

        # no clip in Keras
        clip = None

        if config['go_backwards']:
            direction = "reverse"
        else:
            direction = "forward"

        hidden_size = config['units']

        output_sequence = 0
        if config['return_sequences']:
            output_sequence = 1

        # onnx input
        symbolic_weights = layer.weights
        weights_values = K.batch_get_value(symbolic_weights)

        W_name = symbolic_weights[0].name
        W_weight = weights_values[0]
        graph_input_list.append(cls.make_symbolic_weights(W_name, W_weight))
        weight_list.append(cls.make_weights(W_name, W_weight))

        R_name = symbolic_weights[1].name
        R_weight = weights_values[1]
        graph_input_list.append(cls.make_symbolic_weights(R_name, R_weight))
        weight_list.append(cls.make_weights(R_name, R_weight))

        X = layer.input.name
        inputs = [X, W_name, R_name]

        if config['use_bias']:
            B_name = symbolic_weights[2].name
            B_weight = weights_values[2]
            B_weight = np.concatenate(
                (B_weight[:hidden_size], B_weight[3 * hidden_size:], B_weight[hidden_size:3 * hidden_size]), axis=0)
            graph_input_list.append(cls.make_symbolic_weights(B_name, B_weight))
            weight_list.append(cls.make_weights(B_name, B_weight))

            inputs.append(B_name)
        else:
            inputs.append("")

        # these inputs are not used
        sequence_lens = None

        # onnx output
        # Y represent all intermediate value
        # Y should used when return_sequences is true
        # Y_h represent last output value of hidden
        # set by return_sequence

        Y = ""
        Y_h = ""

        if output_sequence == 1:
            Y = layer.output.name
        else:
            Y_h = layer.output.name
        outputs = [Y, Y_h]

        node_list.append(make_node("RNN",
                                   inputs=inputs,
                                   outputs=outputs,
                                   name=layer.name,
                                   activations=activations,
                                   activation_alpha=activation_alpha,
                                   activation_beta=activation_beta,
                                   direction=direction,
                                   hidden_size=hidden_size,
                                   output_sequence=output_sequence))

        return graph_input_list, weight_list, node_list

    @classmethod
    def create_dense(cls, layer):

        config = layer.get_config()

        node_list = []
        graph_input_list = []
        weight_list = []

        # onnx attribute
        # using default value

        alpha = 1.0
        beta = 1.0
        broadcast = 0
        transA = 0
        transB = 0

        symbolic_weights = layer.weights
        weights_values = K.batch_get_value(symbolic_weights)

        W_name = symbolic_weights[0].name
        B_name = symbolic_weights[1].name

        W_weight, B_weight = weights_values

        graph_input_list.append(cls.make_symbolic_weights(W_name, W_weight))
        graph_input_list.append(cls.make_symbolic_weights(B_name, B_weight))

        weight_list.append(cls.make_weights(W_name, W_weight))
        weight_list.append(cls.make_weights(B_name, B_weight))

        # onnx inputs
        X_name = layer.input.name
        inputs = [X_name, W_name, B_name]

        # onnx outputs
        Y_name = layer.output.name
        outputs = [Y_name]

        if config['activation'] == "linear":
            node = make_node("Gemm",
                             inputs=inputs,
                             outputs=outputs,
                             name=layer.name,
                             alpha=alpha,
                             beta=beta,
                             broadcast=broadcast,
                             transA=transA,
                             transB=transB)
            node_list.append(node)
        else:
            dense_output_name = layer.name + "_output"
            node = make_node("Gemm",
                             inputs=inputs,
                             outputs=[dense_output_name],
                             name=layer.name,
                             alpha=alpha,
                             beta=beta,
                             broadcast=broadcast,
                             transA=transA,
                             transB=transB)
            extra_node = cls.create_extra_activation(layer, dense_output_name)
            node_list.extend([node, extra_node])

        return graph_input_list, weight_list, node_list

    @classmethod
    def create_extra_activation(cls, layer, new_input):
        # layer is origin keras layer
        # new_input is construct new input name for extra activation
        config = layer.get_config()

        activation_node = make_node(rename_operator[config['activation']],
                                    inputs=[new_input],
                                    outputs=[layer.output.name],
                                    name=layer.name + '/' + config['activation'])

        return activation_node

    @classmethod
    def create_conv(cls, layer, dims):

        config = layer.get_config()

        node_list = []
        graph_input_list = []
        weight_list = []

        # current, the order of weights is : [kernel, bias]
        symbolic_weights = layer.weights
        weights_values = K.batch_get_value(symbolic_weights)

        kernel_name = symbolic_weights[0].name
        kernel_weight = weights_values[0]
        # convert kernel shape to onnx
        # [(kernel shape), channels, filters] -> [filters, channels, (kernel shape)]
        dims = list(range(np.ndim(kernel_weight)))
        kernel_weight = np.transpose(kernel_weight, axes=dims[-2:][::-1] + dims[:-2])
        graph_input_list.append(cls.make_symbolic_weights(kernel_name, kernel_weight))

        weight_list.append(make_tensor(name=kernel_name,
                                       data_type=NP_TYPE_TO_TENSOR_TYPE[kernel_weight.dtype],
                                       dims=kernel_weight.shape,
                                       vals=kernel_weight.flatten().tolist()))
        # onnx node inputs
        inputs = [layer.input.name, kernel_name]

        if config['use_bias']:
            bias_name = symbolic_weights[1].name
            kernel_weight, bias_weight = weights_values
            graph_input_list.append(cls.make_symbolic_weights(bias_name, bias_weight))
            weight_list.append(make_tensor(name=bias_name,
                                           data_type=NP_TYPE_TO_TENSOR_TYPE[bias_weight.dtype],
                                           dims=bias_weight.shape,
                                           vals=bias_weight.flatten().tolist()))
            inputs.append(bias_name)

        # get onnx attribute from keras config
        strides = list(config['strides'])
        auto_pad = "SAME_UPPER"
        if config['padding'] == "valid":
            auto_pad = "VALID"
        kernel_shape = list(config['kernel_size'])
        dilations = list(config['dilation_rate'])
        # TODO need a function to convert same or valid into pads
        # pads = []

        # onnx node outputs
        outputs = [layer.output.name]

        if config['activation'] == 'linear':
            node = make_node(
                "Conv",
                inputs=inputs,
                outputs=outputs,
                name=layer.name,
                auto_pad=auto_pad,
                kernel_shape=kernel_shape,
                dilations=dilations,
                strides=strides
            )
            node_list.append(node)
        else:
            # this Conv layer contain a activation function
            # add a extra activation node
            conv_outputs_name = layer.name + "_conv_output"
            conv_node = make_node("Conv",
                                  inputs=inputs,
                                  outputs=[conv_outputs_name],
                                  name=layer.name,
                                  auto_pad=auto_pad,
                                  kernel_shape=kernel_shape,
                                  dilations=dilations,
                                  strides=strides)
            extra_node = cls.create_extra_activation(layer, conv_outputs_name)
            node_list.extend([conv_node, extra_node])

        return graph_input_list, weight_list, node_list

    @classmethod
    def create_conv_transpose(cls, layer, dims):
        if dims != 2:
            raise NotImplementedError("ConvTranspose only support 2D in keras")

        config = layer.get_config()

        node_list = []
        graph_input_list = []
        weight_list = []

        # current, the order of weights is : [kernel, bias]
        symbolic_weights = layer.weights
        weights_values = K.batch_get_value(symbolic_weights)

        kernel_name = symbolic_weights[0].name
        kernel_weight = weights_values[0]
        graph_input_list.append(cls.make_symbolic_weights(kernel_name, kernel_weight))
        # convert kernel shape to onnx
        # [(kernel shape), channels, filters] -> [filters, channels, (kernel shape)]
        dims = list(range(np.ndim(kernel_weight)))
        kernel_weight = np.transpose(kernel_weight, axes=dims[-2:][::-1] + dims[:-2])
        weight_list.append(cls.make_weights(kernel_name, kernel_weight))

        inputs = [layer.input.name, kernel_name]

        if config['use_bias']:
            bias_name = symbolic_weights[1].name
            bias_weight = weights_values[1]
            graph_input_list.append(cls.make_symbolic_weights(bias_name, bias_weight))
            weight_list.append(cls.make_weights(bias_name, bias_weight))
            inputs.append(bias_name)

        # get onnx attribute from keras config
        strides = list(config['strides'])
        auto_pad = "SAME_UPPER"
        if config['padding'] == "valid":
            auto_pad = "VALID"
        kernel_shape = list(config['kernel_size'])
        # TODO need a function to convert same or valid into pads
        # pads = []

        outputs = [layer.output.name]

        if config['activation'] == 'linear':
            node = make_node(
                "ConvTranspose",
                inputs=inputs,
                outputs=outputs,
                name=layer.name,
                auto_pad=auto_pad,
                kernel_shape=kernel_shape,
                strides=strides
            )
            node_list.append(node)
        else:
            # this Conv layer contain a activation function
            # add a extra activation node
            conv_outputs_name = layer.name + "_conv_trans_output"
            conv_node = make_node("ConvTranspose",
                                  inputs=inputs,
                                  outputs=[conv_outputs_name],
                                  name=layer.name,
                                  auto_pad=auto_pad,
                                  kernel_shape=kernel_shape,
                                  strides=strides)
            extra_node = cls.create_extra_activation(layer, conv_outputs_name)
            node_list.extend([conv_node, extra_node])

        return graph_input_list, weight_list, node_list

    @classmethod
    def create_cropping(cls, layer, dims):
        if dims != 2:
            raise NotImplementedError("cropping only support 2D in onnx")

        config = layer.get_config()
        cropping = config["cropping"]

        node_list = []
        graph_input_list = []
        weight_list = []

        # onnx attribute
        # order of border: leftBorder, topBorder, rightBorder, bottomBorder
        border = [cropping[0][0], cropping[1][1], cropping[0][1], cropping[1][1]]

        inputs = [layer.input.name]
        outputs = [layer.output.name]

        node = make_node("Crop",
                         inputs=inputs,
                         outputs=outputs,
                         name=layer.name,
                         border=border)
        node_list.append(node)

        return graph_input_list, weight_list, node_list

    @classmethod
    def create_upsampling(cls, layer, dims):
        if dims != 2:
            raise NotImplementedError("Upsampling only support 2D in onnx")
        config = layer.get_config()

        node_list = []
        graph_input_list = []
        weight_list = []

        # onnx attribute

        width_scale = config['size'][0]
        height_scale = config['size'][1]

        # onnx inputs
        inputs = [layer.input.name]
        outputs = [layer.output.name]

        node = make_node("Upsample",
                         inputs=inputs,
                         outputs=outputs,
                         name=layer.name,
                         width_scale=width_scale,
                         height_scale=height_scale,
                         mode="nearest")
        node_list.append(node)

        return graph_input_list, weight_list, node_list

    @classmethod
    def create_zero_padding(cls, layer, dims):
        node_list = []
        graph_input_list = []
        weight_list = []

        config = layer.get_config()

        # padding in Keras, is [(begin,end),(begin,end),...,]
        padding = config["padding"]

        # onnx attribute
        mode = "constant"
        # however, pads, in onnx ,is like [begin0, begin1, begin2, ... ,end0, end1]
        pads = np.asarray(padding).transpose().flatten().tolist()
        value = 0.0

        # onnx input
        inputs = [layer.input.name]
        outputs = [layer.output.name]

        node = make_node("Pad",
                         inputs=inputs,
                         outputs=outputs,
                         mode=mode,
                         pads=pads,
                         value=value)
        node_list.append(node)

        return graph_input_list, weight_list, node_list

    @classmethod
    def create_max_pooling(cls, layer, dims):
        node_list = []
        graph_input_list = []
        weight_list = []

        # onnx attribute

        config = layer.get_config()
        strides = list(config['strides'])

        auto_pad = "SAME_UPPER"
        if config['padding'] == "valid":
            auto_pad = "VALID"

        pads = []  # need a function to convert same or valid into pads?
        pool_shape = list(config['pool_size'])

        # onnx inputs
        inputs = [layer.input.name]
        outputs = [layer.output.name]

        node = make_node("MaxPool",
                         inputs=inputs,
                         outputs=outputs,
                         name=layer.name,
                         auto_pad=auto_pad,
                         kernel_shape=pool_shape,
                         strides=strides)
        node_list.append(node)

        return graph_input_list, weight_list, node_list

    @classmethod
    def create_average_pooling(cls, layer, dims):
        node_list = []
        graph_input_list = []
        weight_list = []

        # onnx attribute

        config = layer.get_config()
        strides = list(config['strides'])

        auto_pad = "SAME_UPPER"
        if config['padding'] == "valid":
            auto_pad = "VALID"

        pads = []  # need a function to convert same or valid into pads?
        pool_shape = list(config['pool_size'])

        inputs = [layer.input.name]
        outputs = [layer.output.name]

        node = make_node("AveragePool",
                         inputs=inputs,
                         outputs=outputs,
                         name=layer.name,
                         auto_pad=auto_pad,
                         kernel_shape=pool_shape,
                         strides=strides)
        node_list.append(node)
        return graph_input_list, weight_list, node_list

    @classmethod
    def create_global_average_pooling(cls, layer, dims):
        node_list = []
        graph_input_list = []
        weight_list = []

        inputs = [layer.input.name]
        outputs = [layer.output.name]

        node = make_node("GlobalAveragePool",
                         inputs=inputs,
                         outputs=outputs,
                         name=layer.name)
        node_list.append(node)

        return graph_input_list, weight_list, node_list

    @classmethod
    def create_global_max_pooling(cls, layer, dims):
        node_list = []
        graph_input_list = []
        weight_list = []

        inputs = [layer.input.name]
        outputs = [layer.output.name]

        node = make_node("GlobalMaxPool",
                         inputs=inputs,
                         outputs=outputs,
                         name=layer.name)
        node_list.append(node)

        return graph_input_list, weight_list, node_list

    @classmethod
    def create_conv1D(cls, layer):
        return cls.create_conv(layer, 1)

    @classmethod
    def create_conv2D(cls, layer):
        return cls.create_conv(layer, 2)

    @classmethod
    def create_conv3D(cls, layer):
        return cls.create_conv(layer, 3)

    @classmethod
    def create_conv2D_transpose(cls, layer):
        return cls.create_conv_transpose(layer, 2)

    @classmethod
    def create_cropping1D(cls, layer):
        return cls.create_cropping(layer, 1)

    @classmethod
    def create_cropping2D(cls, layer):
        return cls.create_cropping(layer, 2)

    @classmethod
    def create_cropping3D(cls, layer):
        return cls.create_cropping(layer, 3)

    @classmethod
    def create_upsampling1D(cls, layer):
        return cls.create_upsampling(layer, 1)

    @classmethod
    def create_upsampling2D(cls, layer):
        return cls.create_upsampling(layer, 2)

    @classmethod
    def create_upsampling3D(cls, layer):
        return cls.create_upsampling(layer, 3)

    @classmethod
    def create_zero_padding1D(cls, layer):
        return cls.create_zero_padding(layer, 1)

    @classmethod
    def create_zero_padding2D(cls, layer):
        return cls.create_zero_padding(layer, 2)

    @classmethod
    def create_zero_padding3D(cls, layer):
        return cls.create_zero_padding(layer, 3)

    @classmethod
    def create_max_pooling1D(cls, layer):
        return cls.create_max_pooling(layer, 1)

    @classmethod
    def create_max_pooling2D(cls, layer):
        return cls.create_max_pooling(layer, 2)

    @classmethod
    def create_max_pooling3D(cls, layer):
        return cls.create_max_pooling(layer, 3)

    @classmethod
    def create_average_pooling1D(cls, layer):
        return cls.create_average_pooling(layer, 1)

    @classmethod
    def create_average_pooling2D(cls, layer):
        return cls.create_average_pooling(layer, 2)

    @classmethod
    def create_average_pooling3D(cls, layer):
        return cls.create_average_pooling(layer, 2)

    @classmethod
    def create_global_max_pooling1D(cls, layer):
        return cls.create_global_max_pooling(layer, 1)

    @classmethod
    def create_global_max_pooling2D(cls, layer):
        return cls.create_global_average_pooling(layer, 2)

    @classmethod
    def create_global_average_pooling1D(cls, layer):
        return cls.create_global_average_pooling(layer, 1)

    @classmethod
    def create_global_average_pooling2D(cls, layer):
        return cls.create_global_average_pooling(layer, 2)

    @classmethod
    def create_dropout(cls, layer):
        config = layer.get_config()
        node_list = []
        graph_input_list = []
        weight_list = []

        node = make_node("Dropout",
                         inputs=[layer.input.name],
                         outputs=[layer.output.name],
                         name=layer.name,
                         ratio=config['rate'],
                         is_test=1)
        node_list.append(node)
        return graph_input_list, weight_list, node_list

    @classmethod
    def create_reshape(cls, layer):
        config = layer.get_config()
        node_list = []
        graph_input_list = []
        weight_list = []

        # onnx input

        shape_weights = np.array(config['target_shape'])
        shape_name = layer.name + "/shape"
        weight_list.append(cls.make_weights(shape_name, shape_weights))
        graph_input_list.append(cls.make_symbolic_weights(shape_name, shape_weights))

        node = make_node("Reshape",
                         inputs=[layer.input.name],
                         outputs=[layer.output.name],
                         name=layer.name,
                         shape=shape_weights.tolist())
        node_list.append(node)
        return graph_input_list, weight_list, node_list

    # in ONNX, axis = 0 mean flatten all axis into one
    # which is exactly meaning of flatten in Keras
    @classmethod
    def create_flatten(cls, layer):
        node_list = []
        graph_input_list = []
        weight_list = []
        node = make_node("Flatten",
                         inputs=[layer.input.name],
                         outputs=[layer.output.name],
                         name=layer.name,
                         axis=0)
        node_list.append(node)
        return graph_input_list, weight_list, node_list

    # following is basic activation layer

    # in Keras softmax apply to last dimension
    # input is like (nb_samples, nb_timestpes, nb_dims)
    # or (nb_samples, nb_dims)

    # in ONNX , there is a attribute to specify axis
    @classmethod
    def create_softmax(cls, layer):
        node_list = []
        graph_input_list = []
        weight_list = []
        node = make_node("Softmax",
                         inputs=[layer.input.name],
                         outputs=[layer.output.name],
                         name=layer.name,
                         axis=1)
        node_list.append(node)
        return graph_input_list, weight_list, node_list

    # TODO: in ONNX there is a alpha attribute
    @classmethod
    def create_elu(cls, layer):
        node_list = []
        graph_input_list = []
        weight_list = []
        node = make_node("Elu",
                         inputs=[layer.input.name],
                         outputs=[layer.output.name],
                         name=layer.name,
                         alpha=1)
        node_list.append(node)
        return graph_input_list, weight_list, node_list

    # TODO: in ONNX there is alpha and gamma attribute
    @classmethod
    def create_selu(cls, layer):
        node_list = []
        graph_input_list = []
        weight_list = []
        node = make_node("Selu",
                         inputs=[layer.input.name],
                         outputs=[layer.output.name],
                         name=layer.name, )
        node_list.append(node)
        return graph_input_list, weight_list, node_list

    @classmethod
    def create_softplus(cls, layer):
        node_list = []
        graph_input_list = []
        weight_list = []
        node = make_node("Softplus",
                         inputs=[layer.input.name],
                         outputs=[layer.output.name],
                         name=layer.name)
        node_list.append(node)
        return graph_input_list, weight_list, node_list

    def create_softsign(cls, layer):
        node_list = []
        graph_input_list = []
        weight_list = []
        node = make_node("SoftSign",
                         inputs=[layer.input.name],
                         outputs=[layer.output.name],
                         name=layer.name)
        node_list.append(node)
        return graph_input_list, weight_list, node_list

    @classmethod
    def create_relu(cls, layer):
        node_list = []
        graph_input_list = []
        weight_list = []
        node = make_node("Relu",
                         inputs=[layer.input.name],
                         outputs=[layer.output.name],
                         name=layer.name)
        node_list.append(node)
        return graph_input_list, weight_list, node_list

    @classmethod
    def create_tanh(cls, layer):
        node_list = []
        graph_input_list = []
        weight_list = []
        node = make_node("Tanh",
                         inputs=[layer.input.name],
                         outputs=[layer.output.name],
                         name=layer.name)
        node_list.append(node)
        return graph_input_list, weight_list, node_list

    @classmethod
    def create_sigmoid(cls, layer):
        node_list = []
        graph_input_list = []
        weight_list = []
        node = make_node("Sigmoid",
                         inputs=[layer.input.name],
                         outputs=[layer.output.name],
                         name=layer.name)
        node_list.append(node)
        return graph_input_list, weight_list, node_list

    # TODO: ONNX hard_sogmoid have tow attribute: alpha, beta
    @classmethod
    def create_hard_sigmoid(cls, layer):
        node_list = []
        graph_input_list = []
        weight_list = []
        node = make_node("HardSigmoid",
                         inputs=[layer.input.name],
                         outputs=[layer.output.name],
                         name=layer.name)
        node_list.append(node)
        return graph_input_list, weight_list, node_list

    # following is merge layer:
    # ref: http://keras-cn.readthedocs.io/en/latest/layers/merge/
    @classmethod
    def create_add(cls, layer):
        node_list = []
        graph_input_list = []
        weight_list = []
        node = make_node("Add",
                         inputs=[i.name for i in layer.input],
                         outputs=[layer.output.name],
                         name=layer.name,
                         broadcast=0)
        node_list.append(node)
        return graph_input_list, weight_list, node_list

    @classmethod
    def create_sub(cls, layer):
        node_list = []
        graph_input_list = []
        weight_list = []
        node = make_node("Sub",
                         inputs=[i.name for i in layer.input],
                         outputs=[layer.output.name],
                         name=layer.name,
                         broadcast=0)
        node_list.append(node)
        return graph_input_list, weight_list, node_list

    @classmethod
    def create_mul(cls, layer):
        node_list = []
        graph_input_list = []
        weight_list = []
        node = make_node("Mul",
                         inputs=[i.name for i in layer.input],
                         outputs=[layer.output.name],
                         name=layer.name,
                         broadcast=0)
        node_list.append(node)
        return graph_input_list, weight_list, node_list

    @classmethod
    def create_concate(cls, layer):
        node_list = []
        graph_input_list = []
        weight_list = []
        config = layer.get_config()
        axis = config['axis']
        if K.image_data_format() == "channels_last":
            axis = 1
        node = make_node("Concat",
                         inputs=[i.name for i in layer.input],
                         outputs=[layer.output.name],
                         name=layer.name,
                         axis=axis)
        node_list.append(node)
        return graph_input_list, weight_list, node_list

    @classmethod
    def create_max(cls, layer):
        node_list = []
        graph_input_list = []
        weight_list = []
        node = make_node("Max",
                         inputs=[i.name for i in layer.input],
                         outputs=[layer.output.name],
                         name=layer.name)
        node_list.append(node)
        return graph_input_list, weight_list, node_list

    # following is advanced activation
    # ref: http://keras-cn.readthedocs.io/en/latest/layers/advanced_activation_layer/
    @classmethod
    def create_leaky_relu(cls, layer):
        node_list = []
        graph_input_list = []
        weight_list = []
        config = layer.get_config()
        node = make_node("LeakyRelu",
                         inputs=[layer.input.name],
                         outputs=[layer.output.name],
                         name=layer.name,
                         alpha=config['alpha'])
        node_list.append(node)
        return graph_input_list, weight_list, node_list

    # TODO In ONNX , slope in inputs, where is it in Keras?
    @classmethod
    def create_prelu(cls, layer):
        node_list = []
        graph_input_list = []
        weight_list = []
        config = layer.get_config()
        node = make_node("PRelu",
                         inputs=[layer.input.name],
                         outputs=[layer.output.name],
                         name=layer.name)
        node_list.append(node)
        return graph_input_list, weight_list, node_list

    @classmethod
    def create_threshoulded_relu(cls, layer):
        config = layer.get_config()
        node_list = []
        graph_input_list = []
        weight_list = []
        node = make_node("ThreshouldedRelU",
                         inputs=[layer.input.name],
                         outputs=[layer.output.name],
                         name=layer.name,
                         alpha=config["theta"])
        node_list.append(node)
        return graph_input_list, weight_list, node_list


keras_model_to_onnx_model = KerasFrontend.keras_model_to_onnx_model
