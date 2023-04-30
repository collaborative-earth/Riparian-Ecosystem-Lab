import tensorflow as tf
from keras import Model, layers, models


def build_model(options: dict) -> Model:
    model_type = options["model_type"]
    model_params = options["model_params"]
    if model_type == "resnet":
        model = build_resnet(**model_params)
    elif model_type == "fullyconv":
        model = build_fullyconv(**model_params)
    else:
        raise TypeError(f"Model type: {model_type} not supported")
    return model


def build_resnet(
        num_channels=[16, 32, 64],
        num_blocks=[2, 2, 2],
        l2_reg=None,
        reg=None,
        inits='glorot_uniform',
        last_layer=True,
        input_shape=(256, 256, 4),
        fl_size=7,
        dense=128,
        output_dim=1,
        activation='sigmoid',
        data_augmentation=False
):
    inputs = tf.keras.Input(shape=input_shape)
    if data_augmentation:
        inputs = add_data_augmentation(inputs)
    if isinstance(inits, float):
        weight_inits = tf.keras.initializers.TruncatedNormal(stddev=inits)
    else:
        weight_inits = inits
    if l2_reg is not None:
        l2_reg = tf.keras.regularizers.l2(l2_reg)
    for i, num_ch in enumerate(num_channels):
        # Downscale
        if i == 0:  # pre7  fl_size was 7,at 7 5
            conv_out = tf.keras.layers.Conv2D(num_ch, fl_size, padding='same', input_shape=input_shape,
                                              data_format="channels_last", kernel_initializer=weight_inits,
                                              kernel_regularizer=l2_reg)(inputs)
        else:
            conv_out = tf.keras.layers.Conv2D(num_ch, 3, padding='same', kernel_initializer=weight_inits,
                                              kernel_regularizer=l2_reg)(conv_out)
        conv_out = tf.keras.layers.MaxPool2D(pool_size=[3, 3], padding='same', strides=[2, 2],
                                             data_format="channels_last")(conv_out)

        # Residual blocks
        for j in range(num_blocks[i]):
            # with tf.variable_scope('residual_%d_%d' % (i,j)):
            block_input = conv_out
            conv_out = tf.keras.layers.ReLU()(conv_out)
            conv_out = tf.keras.layers.Conv2D(num_ch, 3, padding='same', kernel_initializer=weight_inits,
                                              kernel_regularizer=l2_reg)(conv_out)
            conv_out = tf.keras.layers.ReLU()(conv_out)
            conv_out = tf.keras.layers.Conv2D(num_ch, 3, padding='same', kernel_initializer=weight_inits,
                                              kernel_regularizer=l2_reg)(conv_out)
            conv_out += block_input
    conv_out = tf.keras.layers.ReLU()(conv_out)
    flattened = tf.keras.layers.Flatten()(conv_out)
    if not last_layer:
        return inputs, flattened
    if reg is None:
        conv_out = tf.keras.layers.Dense(dense, use_bias=True, activation='relu', kernel_initializer=weight_inits,
                                         kernel_regularizer=l2_reg)(flattened)
    else:
        conv_out = tf.keras.layers.Dense(dense, use_bias=True, activation='relu', kernel_initializer=weight_inits,
                                         kernel_regularizer=l2_reg, activity_regularizer=tf.keras.regularizers.l1(reg))(
            flattened)
    conv_out = add_FC_layer(conv_out, output_dim, activation=activation, use_bias=True, inits=inits)
    model = tf.keras.Model(inputs, conv_out)
    return model


def build_fullyconv(
        num_channels=[16, 16, 32, 64],
        num_blocks=[1, 1, 1, 1],
        l2_reg=None,
        reg=None,
        inits='glorot_uniform',
        last_layer=True,
        input_shape=(None, None, 4),
        fl_size=3,
        dense=128,
        output_dim=1,
        activation='sigmoid',
        data_augmentation = False
):
    # will still train with 266x266 as the output of data generator, but can run test images one at a time with dif sizes
    inputs = tf.keras.Input(shape=input_shape)
    if data_augmentation:
        inputs = add_data_augmentation(inputs)
    if isinstance(inits, float):
        weight_inits = tf.keras.initializers.TruncatedNormal(stddev=inits)
    else:
        weight_inits = inits
    if l2_reg is not None:
        l2_reg = tf.keras.regularizers.l2(l2_reg)
    for i, num_ch in enumerate(num_channels):
        # Downscale
        if i == 0:
            conv_out = tf.keras.layers.Conv2D(num_ch, fl_size, padding='same', input_shape=input_shape,
                                              data_format="channels_last", kernel_initializer=weight_inits,
                                              kernel_regularizer=l2_reg)(inputs)
        else:
            conv_out = tf.keras.layers.Conv2D(num_ch, 3, padding='same', kernel_initializer=weight_inits,
                                              kernel_regularizer=l2_reg)(conv_out)
        conv_out = tf.keras.layers.MaxPool2D(pool_size=[3, 3], padding='same', strides=[2, 2],
                                             data_format="channels_last")(conv_out)

        # Residual blocks
        for j in range(num_blocks[i]):
            # with tf.variable_scope('residual_%d_%d' % (i,j)):
            block_input = conv_out
            conv_out = tf.keras.layers.ReLU()(conv_out)
            conv_out = tf.keras.layers.Conv2D(num_ch, 3, padding='same', kernel_initializer=weight_inits,
                                              kernel_regularizer=l2_reg)(conv_out)
            conv_out = tf.keras.layers.ReLU()(conv_out)
            conv_out = tf.keras.layers.Conv2D(num_ch, 3, padding='same', kernel_initializer=weight_inits,
                                              kernel_regularizer=l2_reg)(conv_out)
            conv_out += block_input
    conv_out = tf.keras.layers.ReLU()(conv_out)
    flattened = tf.keras.layers.GlobalAveragePooling2D()(conv_out)
    if not last_layer:
        return inputs, flattened
    if reg is None:
        conv_out = tf.keras.layers.Dense(dense, use_bias=True, activation='relu', kernel_initializer=weight_inits,
                                         kernel_regularizer=l2_reg)(flattened)
    else:
        conv_out = tf.keras.layers.Dense(dense, use_bias=True, activation='relu', kernel_initializer=weight_inits,
                                         kernel_regularizer=l2_reg, activity_regularizer=tf.keras.regularizers.l1(reg))(
            flattened)
    conv_out = add_FC_layer(conv_out, output_dim, activation=activation, use_bias=True, inits=inits)
    model = tf.keras.Model(inputs, conv_out)
    return model


def add_data_augmentation(inputs):
    print("applying data augmentation")
    inputs = layers.RandomFlip("horizontal")(inputs)
    inputs = layers.RandomRotation(0.25)(inputs)
    inputs = layers.RandomTranslation(height_factor=0.1, width_factor=0.1, fill_mode="reflect")(inputs)
    inputs = layers.RandomContrast(factor=0.3)(inputs)
    return inputs


def add_FC_layer(outputs, units, activation=None, use_bias=False, inits='glorot_uniform'):
    if isinstance(inits, float):
        weight_inits = tf.keras.initializers.TruncatedNormal(stddev=inits)
    else:
        weight_inits = inits
    outputs = tf.keras.layers.Dense(units, activation=activation, kernel_initializer=weight_inits, use_bias=use_bias)(
        outputs)
    return outputs


def get_optimizer(opt, learning_rate_fn):
    if opt == 'adadelta':
        optimizer = tf.keras.optimizers.Adadelta(learning_rate=learning_rate_fn)
    elif opt == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn)
    elif opt == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate_fn)
    elif opt == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate_fn)
    elif opt == 'adagrad':
        optimizer = tf.keras.optimizers.Adagrad(learning_rate=learning_rate_fn)
    return optimizer


def load_pretrained_model(pretrained_model_path):
    model = models.load_model(pretrained_model_path)
    return model