import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv2D, Conv2DTranspose, Reshape, ReLU, BatchNormalization, LeakyReLU, Flatten, Dense
from keras.models import Model, Sequential


def specgan_generator(latent_dim=64):
    model = Sequential([
        Dense(4 * 4 * latent_dim * 16),
        Reshape((4, 4, latent_dim * 16)),
        BatchNormalization(),
        ReLU(),

        Conv2DTranspose(filters=latent_dim * 8, kernel_size=5,
                        strides=2, padding='same',
                        use_bias=False),
        BatchNormalization(),
        ReLU(),

        Conv2DTranspose(filters=latent_dim * 4, kernel_size=5,
                        strides=2, padding='same',
                        use_bias=False),
        BatchNormalization(),
        ReLU(),

        Conv2DTranspose(filters=latent_dim * 2, kernel_size=5,
                        strides=2, padding='same',
                        use_bias=False),
        BatchNormalization(),
        ReLU(),

        Conv2DTranspose(filters=latent_dim, kernel_size=5,
                        strides=2, padding='same',
                        use_bias=False),
        BatchNormalization(),
        ReLU(),

        Conv2DTranspose(filters=1, kernel_size=5,
                        strides=2, padding='same',
                        use_bias=False),
        keras.layers.Activation('sigmoid')
    ])

    return model


def specgan_discriminator(latent_dim=64):
    model = Sequential([
        Conv2D(filters=latent_dim, kernel_size=5,
               strides=2, padding='same',
               use_bias=False),
        BatchNormalization(),
        LeakyReLU(0.2),

        Conv2D(filters=latent_dim * 2, kernel_size=5,
               strides=2, padding='same',
               use_bias=False),
        BatchNormalization(),
        LeakyReLU(0.2),

        Conv2D(filters=latent_dim * 4, kernel_size=5,
               strides=2, padding='same',
               use_bias=False),
        BatchNormalization(),
        LeakyReLU(0.2),

        Conv2D(filters=latent_dim * 8, kernel_size=5,
               strides=2, padding='same',
               use_bias=False),
        BatchNormalization(),
        LeakyReLU(0.2),

        Conv2D(filters=latent_dim * 16, kernel_size=5,
               strides=2, padding='same',
               use_bias=False),
        BatchNormalization(),
        LeakyReLU(0.2),

        Flatten(),
        Dense(1, activation='sigmoid'),
    ])

    return model


if __name__ == "__main__":
    gen = specgan_generator()
    gen.build((100, 64))
    gen.summary()

    disc = specgan_discriminator()
    disc.build((100, 128, 128, 1))
    disc.summary()
