import tensorflow as tf
import tensorflow_probability as tfp

tfl = tf.keras.layers
tfd = tfp.distributions


class VampVAE(tf.keras.Model):

    def __init__(self, latents=32, inducing_inputs=500, name="vamp_prior_vae", **kwargs):
        super(VampVAE, self).__init__(name=name, **kwargs)

        self.latents = latents

        self.encoder = Encoder(latents=self.latents)
        self.decoder = Decoder(latents=self.latents)

        self.num_inducing_inputs = inducing_inputs
        self.inducing_inputs = tf.Variable(tf.random.uniform(shape=(inducing_inputs, 28, 28, 1),
                                                             minval=0.2,
                                                             maxval=0.8),
                                           name="inducing_inputs")

    def call(self, tensor, training=True):
        loc, scale = self.encoder(tensor)

        self.prior = tfd.Mixture(cat=tfd.Categorical(probs=[[1 / self.num_inducing_inputs] * self.num_inducing_inputs] * self.latents),
                                 components=[tfd.Normal(loc=l, scale=s) for l, s in
                                             zip(*self.encoder(self.inducing_inputs))])

        self.posterior = tfd.Normal(loc=loc, scale=scale)

        code = self.posterior.sample()

        reconstruction = self.decoder(code)

        self.likelihood = tfd.Normal(loc=tensor,
                                     scale=tf.ones_like(tensor))

        return reconstruction


class MFVAE(tf.keras.Model):

    def __init__(self, latents=32, name="mean_field_vae", **kwargs):
        super(MFVAE, self).__init__(name=name, **kwargs)

        self.latents = latents

        self.encoder = Encoder(latents=self.latents)
        self.decoder = Decoder(latents=self.latents)

    def call(self, tensor, training=True):
        loc, scale = self.encoder(tensor)

        self.prior = tfd.Normal(loc=tf.zeros_like(loc),
                                scale=tf.ones_like(scale))

        self.posterior = tfd.Normal(loc=loc,
                                    scale=scale)

        code = self.posterior.sample()

        reconstruction = self.decoder(code)

        self.likelihood = tfd.Normal(loc=tensor,
                                     scale=tf.ones_like(tensor))

        return reconstruction


class Encoder(tfl.Layer):

    def __init__(self, latents=32, name="encoder", **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)

        self.latents = latents

        self.layers = []

    def build(self, input_size):
        # 28 x 28 x 1 ->

        self.layers = [
            tfl.Conv2D(filters=4,
                       kernel_size=(5, 5),
                       strides=(1, 1),
                       padding="same"),
            tf.nn.relu,
            tfl.Conv2D(filters=16,
                       kernel_size=(3, 3),
                       strides=(2, 2),
                       padding="same"),
            tf.nn.relu,
            tfl.Conv2D(filters=64,
                       kernel_size=(3, 3),
                       strides=(2, 2),
                       padding="same"),
            tf.nn.relu,
            tfl.Flatten(),
            tfl.Dense(units=128),
            tf.nn.relu
        ]

        self.loc_head = tfl.Dense(units=self.latents)
        self.scale_head = tfl.Dense(units=self.latents, activation="softplus")

        super(Encoder, self).build(input_size)

    def call(self, tensor, training=True):
        for layer in self.layers:
            tensor = layer(tensor)

        loc = self.loc_head(tensor)
        scale = self.scale_head(tensor)

        return loc, scale


class Decoder(tfl.Layer):

    def __init__(self, latents=32, name="decoder", **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)

        self.latents = latents

        self.layers = []

    def build(self, input_size):
        # 28 x 28 x 1 ->

        self.layers = [
            tfl.Dense(units=128),
            tf.nn.relu,
            tfl.Dense(units=64 * 7 * 7),
            tf.nn.relu,
            tfl.Reshape(target_shape=(7, 7, 64)),
            tfl.Conv2DTranspose(filters=16,
                                kernel_size=(3, 3),
                                strides=(2, 2),
                                padding="same"),
            tf.nn.relu,
            tfl.Conv2DTranspose(filters=4,
                                kernel_size=(3, 3),
                                strides=(2, 2),
                                padding="same"),
            tf.nn.relu,
            tfl.Conv2DTranspose(filters=1,
                                kernel_size=(5, 5),
                                strides=(1, 1),
                                padding="same"),
            tf.nn.sigmoid
        ]

        super(Decoder, self).build(input_size)

    def call(self, tensor, training=True):
        for layer in self.layers:
            tensor = layer(tensor)

        return tensor
