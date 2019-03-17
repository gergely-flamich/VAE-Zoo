import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import sonnet as snt
import numpy as np
from functools import reduce

class VAE(snt.AbstractModule):

    def __init__(self, input_shape, num_units, latent_dim=2, name="vae"):

        super(VAE, self).__init__(name=name)

        self.input_shape = input_shape
        self.num_inputs = reduce(lambda x, y: x * y, input_shape)
        self.num_units = num_units
        self.latent_dim = latent_dim
        self.q_distribution = None
        self.latent_prior = None


    def kl_divergence(self):
        """
        Calculates the KL divergence between the current variational posterior and the prior:

        KL[ q(z | theta) || p(z) ]

        """
        if self.q_distribution is None or self.latent_prior is None:
            raise Exception("VAE module needs to be connected into the graph before calculating the KL divergence of the variational posterior and the prior!")
        return tf.reduce_sum(tfd.kl_divergence(self.q_distribution, self.latent_prior))


    def input_log_prob(self):
        """
        Returns the log-likelihood of the current input for the output Bernoulli
        """
        return tf.reduce_sum(self.log_prob)


    @snt.reuse_variables
    def encode(self, inputs):
        """
        Builds the encoder part of the VAE, i.e. q(x | theta).
        This maps from the input to the latent representation.
        """
        flatten = snt.BatchFlatten()
        flattened = flatten(inputs)

        # Create hidden layers
        linear = snt.Linear(self.num_units,
                            name='encoder_hidden')
        dense = linear(flattened)
        dense = tf.nn.relu(dense)

        # Mean for the latent distributions
        linear = snt.Linear(self.latent_dim,
                            name='encoder_mu')
        mu = linear(dense)

        # Standard deviation for the latent distributions
        # The softplus will ensure that sigma is positive
        linear = snt.Linear(self.latent_dim,
                            name='encoder_sigma')

        sigma = linear(dense)
        sigma = tf.nn.softplus(sigma)

        return mu, sigma


    @snt.reuse_variables
    def decode(self, latent_code):
        """
        Builds the decoder part of the VAE
        """

        # Create regular hidden layers
        # Layer 1
        linear = snt.Linear(self.num_units,
                            name='decoder_hidden_1')
        dense = linear(latent_code)
        dense = tf.nn.relu(dense)

        # Layer 2
        linear = snt.Linear(self.num_units,
                            name='decoder_hidden_2')
        dense = linear(dense)
        dense = tf.nn.relu(dense)

        # Layer 3
        linear = snt.Linear(self.num_inputs,
                            name='decoder_hidden_logits')
        logits = linear(dense)

        to_output_shape = snt.BatchReshape(shape=self.input_shape)
        output = to_output_shape(logits)

        decoder_bernoulli = tfd.Bernoulli(logits=output)

        return decoder_bernoulli


    def _build(self, inputs):
        """
        Build standard VAE:
        1. Encode input -> latent mu, sigma
        2. Sample z ~ N(z | mu, sigma)
        3. Decode z -> output Bernoulli means
        4. Sample o ~ Bernoulli(o | z)
        """
        #assert tf.shape(inputs) == self.input_shape

        # Define the prior
        self.latent_prior = tfd.Normal(
            loc=np.zeros(self.latent_dim, dtype=np.float32),
            scale=np.ones(self.latent_dim, dtype=np.float32))

        # Define the encoding - decoding process
        q_mu, q_sigma = self.encode(inputs)

        self.q_distribution = tfd.Normal(loc=q_mu, scale=q_sigma)
        assert self.q_distribution.reparameterization_type == tfd.FULLY_REPARAMETERIZED

        z = self.q_distribution.sample()

        output_distribution = self.decode(z)

        self.log_prob = output_distribution.log_prob(inputs)

        return output_distribution.sample()


