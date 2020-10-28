import tensorflow as tf

num_filters = 3
num_convs = 3

def conv_fn(conv_in):
    """
    Convnet feature extractor from the state observations.

    Args:
        conv_in:  B x H x W x D

    Returns:
        B x Dout matrix
    """
    assert len(conv_in.shape) == 4
    conv_out = tf.layers.conv2d(
        inputs=conv_in,
        filters=num_filters,
        kernel_size=[2, 2],
        padding="same",
        activation=tf.nn.leaky_relu,
        name="conv_initial"
    )

    for i in range(0, num_convs - 1):
        padding = "same" if i < num_convs - 2 else "valid"
        conv_out = tf.layers.conv2d(
            inputs=conv_out,
            filters=num_filters,
            kernel_size=[2, 2],
            padding=padding,
            activation=tf.nn.leaky_relu,
            name="conv_{}".format(i)
        )
    out = tf.layers.flatten(conv_out)
    assert len(out.shape) == 2, '{}'.format(out.shape)
    return out

T = 10
x = tf.random_normal(shape=[10, T, 32, 32, 3])
_input = x[:,0,:,:]

batchsize, h_length, height, width, depth = x.shape
x = tf.reshape(x, shape=[batchsize * h_length, height, width, depth])

out = conv_fn(x)
out = conv_fn(_input)


for x in tf.global_variables():
    print(x.name)
