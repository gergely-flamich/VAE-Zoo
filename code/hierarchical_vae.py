from functools import reduce
from numpy import prod

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from sonnet import AbstractModule, Linear, Conv2D, BatchFlatten, BatchNorm, BatchReshape, reuse_variables


class HierarchicalVAE(AbstractModule):

    _allowed_latent_dists = {
        "gaussian": tfd.Normal,
        "laplace": tfd.Laplace
    }

    _allowed_likelihoods = {
        "gaussian": tfd.Normal,
        "laplace": tfd.Laplace
    }

    _latent_priors = []
    _latent_posteriors = []

    def __init__(self,
                 hidden_sizes,
                 latent_sizes,
                 output_size,
                 latent_dist="gaussian",
                 likelihood="gaussian",
                 standardized=False,
                 name="hierarchical_vae"):

        super(HierarchicalVAE, self).__init__(name=name)

        if type(hidden_sizes) != tuple:
            raise tf.errors.InvalidArgumentError("hidden_sizes must be a tuple!")

        if type(latent_sizes) != tuple:
            raise tf.errors.InvalidArgumentError("latent_sizes must be a tuple!")

        if type(output_size) != tuple:
            raise tf.errors.InvalidArgumentError("output_size must be a tuple!")

        if len(hidden_sizes) != len(latent_sizes):
            raise tf.errors.InvalidArgumentError("hidden_sizes and latent_sizes"
                                                 " must be the same lenght! {} ~= {}".format(
                                                     len(hidden_sizes),
                                                     len(latent_sizes)))

        self._latent_sizes = latent_sizes
        self._hidden_sizes = hidden_sizes
        self._output_size = output_size
        self._num_levels = len(latent_sizes)

        if latent_dist not in self._allowed_latent_dists:
            raise tf.errors.InvalidArgumentError("latent_dist must be one of {}"
                                                 .format(self._allowed_latent_dists))

        self._latent_dist = self._allowed_latent_dists[latent_dist]

        if likelihood not in self._allowed_likelihoods:
            raise tf.errors.InvalidArgumentError("likelihood must be one of {}"
                                                 .format(self._allowed_likelihoods))

        self._likelihood_dist = self._allowed_likelihoods[likelihood]

        self._standardized = standardized


    @property
    def kl_divergence(self):
        self._ensure_is_connected()

        if (len(self._latent_posteriors) != self._num_levels or
            len(self._latent_priors) != self._num_levels):

            raise Exception("Need a full pass through the VAE to calculate KL!")

        klds = [tf.reduce_sum(tfd.kl_divergence(posterior, prior))
                for posterior, prior in zip(self._latent_posteriors, self._latent_priors)]

        return sum(klds)

    @property
    def input_log_prob(self):
        self._ensure_is_connected()

        if (len(self._latent_posteriors) != self._num_levels or
            len(self._latent_priors) != self._num_levels):

            raise Exception("Need a full pass through the VAE to calculate log probability!")

        return tf.reduce_sum(self._log_prob)


    @reuse_variables
    def training_finished(self):

        for layer in self._encoding_layers:
            layer.training_finished()

        for layer in self._decoding_layers:
            layer.training_finished()


    @reuse_variables
    def encode(self, inputs, level=1):

        # ----------------------------------------------------------------------
        # Define layers
        # ----------------------------------------------------------------------

        flatten = BatchFlatten()

        self._encoding_layers = [LocScaleVAELayer(hidden_size=hidden_size,
                                            output_size=output_size,
                                            num_hiddens=2,
                                            activation="leaky_relu",
                                            use_batch_norm=True,
                                            name="encoder_level_{}".format(idx))
                           for idx, (hidden_size, output_size)
                           in enumerate(zip(self._hidden_sizes, self._latent_sizes))]

        # ----------------------------------------------------------------------
        # Apply layers
        # ----------------------------------------------------------------------

        # reset the distributions
        self._latent_posteriors = []

        latents = ()

        activations = flatten(inputs)

        for l in range(level):

            loc, scale = self._encoding_layers[l](activations)

            posterior = self._latent_dist(loc=loc, scale=scale)

            self._latent_posteriors.append(posterior)

            activations = posterior.sample()

            latents = latents + (activations,)

        return latents


    @reuse_variables
    def decode(self, latents, level=1):
        # ----------------------------------------------------------------------
        # Define layers
        # ----------------------------------------------------------------------

        self._decoding_layers = [LocScaleVAELayer(hidden_size=hidden_size,
                                            output_size=output_size,
                                            num_hiddens=2,
                                            activation="leaky_relu",
                                            use_batch_norm=True,
                                            name="decoder_level_{}".format(idx))
                           for idx, (hidden_size, output_size)
                           in enumerate(zip(self._hidden_sizes[:0:-1],
                                            self._latent_sizes[-2::-1]))]

        # Add last layer
        self._decoding_layers.append(LocScaleVAELayer(hidden_size=self._hidden_sizes[-1],
                                                output_size=prod(self._output_size),
                                                num_hiddens=2,
                                                activation="leaky_relu",
                                                # No batch norm on last layer
                                                use_batch_norm=False,
                                                name="decoder_out"))

        # ----------------------------------------------------------------------
        # Apply layers
        # ----------------------------------------------------------------------

        # Reverse latents for convenience
        latents = latents[::-1]

        # reset the distributions
        self._latent_priors = [self._latent_dist(loc=tf.zeros_like(latents[0]),
                                                 scale=tf.ones_like(latents[0]))]

        for l in range(level - 1):

            loc, scale = self._decoding_layers[l](latents[l])

            prior = self._latent_dist(loc=loc, scale=scale)

            self._latent_priors.append(prior)


        # Reverse prior list for convenience
        self._latent_priors = self._latent_priors[::-1]

        return self._decoding_layers[level - 1](latents[level - 1])


    def _build(self, inputs):

        latents = self.encode(inputs,
                              level=self._num_levels)

        decoded_loc, decoded_scale = self.decode(latents,
                                                 level=self._num_levels)

        reshaper = BatchReshape(shape=self._output_size)

        decoded_loc = reshaper(decoded_loc)
        decoded_scale = reshaper(decoded_scale)

        likelihood_variance = decoded_scale if not self._standardized else tf.ones_like(decoded_scale)

        self._likelihood = self._likelihood_dist(loc=decoded_loc,
                                                 scale=likelihood_variance)

        self._log_prob = self._likelihood.log_prob(inputs)

        return decoded_loc


class LocScaleVAELayer(AbstractModule):

    _allowed_activations = {
        "relu": tf.nn.relu,
        "leaky_relu": lambda x: tf.nn.leaky_relu(x, alpha=0.1),
        "tanh": tf.nn.tanh,
        "none": tf.identity
    }

    def __init__(self,
                 hidden_size,
                 output_size,
                 num_hiddens=1,
                 activation="relu",
                 use_batch_norm=True,
                 name="loc_scale_vae_layer"):

        super(LocScaleVAELayer, self).__init__(name=name)

        self._num_hiddens = num_hiddens
        self._hidden_size = hidden_size
        self._output_size = output_size
        self._use_batch_norm = use_batch_norm

        if activation not in self._allowed_activations:
            raise tf.errors.InvalidArgumentError("activation must be one of {}"
                                                 .format(self._allowed_activations))

        self._activation = self._allowed_activations[activation]

        self._is_training = True


    @reuse_variables
    def training_finished(self):
        self._is_training = False


    def _build(self, inputs):

        activations = inputs

        # ----------------------------------------------------------------------
        # Define layers
        # ----------------------------------------------------------------------

        self._hidden_layers = [Linear(output_size=self._hidden_size)
                               for i in range(self._num_hiddens)]

        self._loc_head = Linear(output_size=self._output_size)
        self._scale_head = Linear(output_size=self._output_size)

        # ----------------------------------------------------------------------
        # Apply layers
        # ----------------------------------------------------------------------

        for hidden_layer in self._hidden_layers:
            activations = self._activation(hidden_layer(activations))

            if self._use_batch_norm:
                bn = BatchNorm()
                activations = bn(activations, is_training=self._is_training)

        loc = self._loc_head(activations)
        scale = tf.nn.softplus(self._scale_head(activations))

        return loc, scale
