import os
import matplotlib.pyplot as plt
import librosa.display
import tensorflow as tf
import numpy as np
from tensorflow import keras
from model_utils import specgan_discriminator, specgan_generator
from SpecGAN.dataset.preprocess import prepare_dataset

BATCH_SIZE = 32
EPOCHS = 5000


class GAN(keras.Model):
    def __init__(self, latent_dim, disc, gen):
        super(GAN, self).__init__()
        self.g_loss_metric = None
        self.loss_fn = None,
        self.d_loss_metric = None
        self.g_optimizer = None
        self.d_optimizer = None
        self.discriminator = disc
        self.generator = gen
        self.latent_dim = latent_dim

    def compile(self, g_optimizer, d_optimizer, loss_fn, **kwargs):
        super(GAN, self).compile(**kwargs)
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.loss_fn = loss_fn
        self.d_loss_metric = keras.metrics.Mean(name='d_loss')
        self.g_loss_metric = keras.metrics.Mean(name='g_loss')

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    def train_step(self, data):
        random_latent_vector = tf.random.normal(shape=(BATCH_SIZE, self.latent_dim))
        generated_images = self.generator(random_latent_vector)

        combined_images = tf.concat([generated_images, data[0]], axis=0)

        labels = tf.concat([tf.ones(shape=(BATCH_SIZE, 1)), tf.zeros(shape=(BATCH_SIZE, 1))], axis=0)

        real_noise = 0.15 * tf.random.uniform((BATCH_SIZE, 1))
        fake_noise = -0.15 * tf.random.uniform((BATCH_SIZE, 1))
        noise = tf.concat([fake_noise, real_noise], axis=0)

        labels += noise

        # training discriminator
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, predictions)

        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        random_latent_vector = tf.random.normal(shape=(BATCH_SIZE, self.latent_dim))

        misleading_labels = tf.zeros((BATCH_SIZE, 1))

        # training the generator
        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(random_latent_vector))
            g_loss = self.loss_fn(misleading_labels, predictions)

        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(
            zip(grads, self.generator.trainable_weights)
        )

        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)

        return {
            'd_loss': self.d_loss_metric.result(),
            'g_loss': self.g_loss_metric.result()
        }


class GANMonitor(keras.callbacks.Callback):
    def __init__(self, num_img=3, latent_dim=64):
        self.num_img = num_img
        self.latent_dim = latent_dim

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim))
        generated_images = self.model.generator(random_latent_vectors)
        generated_images.numpy()
        os.makedirs("generated/generated_epoch_%d" % epoch, exist_ok=True)
        for i in range(self.num_img):
            spec = generated_images[i].numpy()
            fig, ax = plt.subplots()
            img = librosa.display.specshow(spec[..., 0], x_axis='time',
                                           y_axis='mel', sr=16000,
                                           fmax=8000, ax=ax)
            fig.colorbar(img, ax=ax, format='%+2.0f dB')
            ax.set(title='Mel-frequency spectrogram')
            plt.savefig("generated/generated_epoch_%d/%d.png" % (epoch, i))
            plt.cla()
        plt.close('all')


if __name__ == "__main__":
    dataset = prepare_dataset('/home/nikhil/Downloads/archive/nsynth-train-all/audio')
    discriminator = specgan_discriminator()
    generator = specgan_generator()
    gan = GAN(latent_dim=64, disc=discriminator, gen=generator)

    gan.compile(d_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                g_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                loss_fn=keras.losses.BinaryCrossentropy()
                )
    gan.fit(dataset, epochs=EPOCHS, callbacks=[GANMonitor(num_img=5, latent_dim=64)])
