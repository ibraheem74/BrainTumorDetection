import tensorflow as tf
from tensorflow.keras import layers, Model


def conv_block(inputs, filters):
    x = layers.Conv2D(filters, 3, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    return x


def encoder_block(inputs, filters):
    x = conv_block(inputs, filters)
    p = layers.MaxPooling2D((2, 2))(x)
    return x, p


def attention_gate(g, x, filters):

    g1 = layers.Conv2D(filters, 1, padding="same")(g)
    x1 = layers.Conv2D(filters, 1, padding="same")(x)

    psi = layers.Add()([g1, x1])
    psi = layers.ReLU()(psi)
    psi = layers.Conv2D(1, 1, activation="sigmoid")(psi)

    out = layers.Multiply()([x, psi])

    return out


def decoder_block(inputs, skip, filters):

    x = layers.Conv2DTranspose(
        filters,
        kernel_size=2,
        strides=2,
        padding="same"
    )(inputs)

    skip = attention_gate(x, skip, filters)

    x = layers.Concatenate()([x, skip])

    x = conv_block(x, filters)

    return x


def build_attention_unet(input_shape=(256, 256, 3)):

    inputs = layers.Input(input_shape)

    # Encoder
    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    # Bottleneck
    b = conv_block(p4, 1024)

    # Decoder
    d1 = decoder_block(b, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    outputs = layers.Conv2D(
        1,
        kernel_size=1,
        activation="sigmoid"
    )(d4)

    model = Model(
        inputs,
        outputs,
        name="Attention_U_Net"
    )

    return model