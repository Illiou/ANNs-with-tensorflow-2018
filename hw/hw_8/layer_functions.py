def feed_forward_layer(x, hidden_n, activation_fn, normalize):
    initializer = tf.random_normal_initializer(stddev=0.02)
    weights = tf.get_variable("weights", [x.shape[1], hidden_n], tf.float32, initializer)
    biases = tf.get_variable("biases", [hidden_n], tf.float32, tf.zeros_initializer())
   
    drive = tf.matmul(x, weights) + biases
    if normalize:
        drive = batch_norm(drive, [0])
   
    if activation_fn == 'linear':
        return drive
    else:
        return activation_fn(drive)


def conv_layer(x, kernels_n, kernel_size, stride_size, activation_fn, normalize):
    initializer = tf.random_normal_initializer(stddev=0.02)
    kernels = tf.get_variable("kernels", [kernel_size, kernel_size, x.shape[-1], kernels_n], tf.float32, initializer)
    biases = tf.get_variable("biases", [kernels_n], tf.float32, tf.zeros_initializer())

    drive = tf.nn.conv2d(x, kernels, strides = [1, stride_size, stride_size, 1], padding = "SAME") + biases
    if normalize:
        drive = batch_norm(drive, [0,1,2])
    
    return activation_fn(drive)


def back_conv_layer(x, target_shape, kernel_size, stride_size, activation_fn, normalize):
    initializer = tf.random_normal_initializer(stddev=0.02)
    kernels = tf.get_variable("kernels", [kernel_size, kernel_size, target_shape[-1], x.shape[-1]], tf.float32, initializer)
    biases = tf.get_variable("biases", [target_shape[-1]], tf.float32, tf.zeros_initializer())

    drive = tf.nn.conv2d_transpose(x, kernels, target_shape, strides = [1, stride_size, stride_size, 1], padding = "SAME") + biases
    if normalize:
        drive = batch_norm(drive, [0,1,2])
    
    return activation_fn(drive)


def flatten(x):
    size = int(np.prod(x.shape[1:]))
    return tf.reshape(x, [-1, size])


def batch_norm(x, axes):
    mean, var = tf.nn.moments(x, axes = axes)
    offset_initializer = tf.constant_initializer(0.0)
    offset = tf.get_variable("offset", [x.shape[-1]], tf.float32, offset_initializer)
    scale_initializer = tf.constant_initializer(1.0)
    scale = tf.get_variable("scale", [x.shape[-1]], tf.float32, scale_initializer)
    return tf.nn.batch_normalization(x, mean, var, offset, scale, 1e-6)