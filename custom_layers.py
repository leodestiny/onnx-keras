from keras import backend as K
from keras.engine import InputSpec
from keras.layers.convolutional import _Conv
from keras.layers.core import Layer


class LRN(Layer):
    def __init__(self, alpha=0.0001, k=1, beta=0.75, n=5, **kwargs):
        self.alpha = alpha
        self.k = k
        self.beta = beta
        self.n = n
        super(LRN, self).__init__(**kwargs)

    def call(self, x, mask=None):
        b, ch, r, c = x.shape
        half_n = self.n // 2  # half the local region
        input_sqr = K.square(x)  # square the input

        extra_channels = K.zeros((b, int(ch) + 2 * half_n, r, c))
        input_sqr = K.concatenate(
            [extra_channels[:, :half_n, :, :], input_sqr, extra_channels[:, half_n + int(ch):, :, :]], axis=1)

        scale = self.k  # offset for the scale
        norm_alpha = self.alpha / self.n  # normalized alpha
        for i in range(self.n):
            scale += norm_alpha * input_sqr[:, i:i + int(ch), :, :]
        scale = scale ** self.beta
        x = x / scale
        return x

    def get_config(self):
        config = {"alpha": self.alpha,
                  "k": self.k,
                  "beta": self.beta,
                  "n": self.n}
        base_config = super(LRN, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))


class GroupConv(_Conv):
    def __init__(self, group, **kwargs):
        self.group = group
        super(GroupConv, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (int(input_dim / self.group), self.filters)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: input_dim})
        self.built = True

    def call(self, x):
        if self.group == 1:
            return super(GroupConv, self).call(x)

        else:
            shape = x.shape
            if self.data_format == 'channels_first':
                channel_axis = 1
            else:
                channel_axis = -1

            size = shape[channel_axis] / self.group
            k_size = K.int_shape(self.kernel)[-1] / self.group
            output_list = list()
            for i in range(self.group):
                if self.rank == 2:
                    if self.data_format == 'channels_first':
                        output_list.append(self.cal_conv(
                            x[:, size * i:size * (i + 1), :, :],
                            self.kernel[:, :, :, k_size * i:k_size * (i + 1)]
                        ))

                    else:
                        output_list.append(self.cal_conv(
                            x[:, :, :, size * i:size * (i + 1)],
                            self.kernel[:, :, :, k_size * i:k_size * (i + 1)]
                        ))

            outputs = K.concatenate(output_list, axis=channel_axis)
            if self.use_bias:
                outputs = K.bias_add(outputs, self.bias, data_format=self.data_format)
            return outputs

    def get_config(self):
        config = {"group": self.group}
        base_config = super(GroupConv, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))

    def cal_conv(self, inputs, kernel):
        if self.rank == 1:
            outputs = K.conv1d(
                inputs,
                kernel,
                strides=self.strides[0],
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate[0])
        if self.rank == 2:
            outputs = K.conv2d(
                inputs,
                kernel,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate)
        if self.rank == 3:
            outputs = K.conv3d(
                inputs,
                kernel,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate)
        return outputs
