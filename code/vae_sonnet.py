import tensorflowa as tf
from tensorflow_probability.distributions import Normal, Bernoulli, kl_divergence, FULLY_REPARAMETERIZED
import sonnet as snt

class VAE(snt.AbstractModule):

    def __init__(self, input_shape, num_units, latent_dim=2, name="vae"):

        super(VAE, self).__init__(name)

        self.input_shape = input_shape
        self.num_units = num_units
        self.latent_dim = latent_dim

        self.q_distribution = None
        self.latent_prior = None


    def get_kl_divergence():
        if self.q_distribution is None or self.latent_prior is None:
            raise Exception("VAE module needs to be connected into the graph before calculating the KL divergence of the variational posterior and the prior!")
        return tf.reduce_sum(kl_divergence(self.q_distribution, self.prior))

    @snt.reuse_variables
    def encode(inputs):
        """
        Builds the encoder part of the VAE, i.e. q(x | theta).
        This maps from the input to the latent representation.
        """

        # Create hidden layers
        linear = snt.Linear(self.num_units,
                            name='encoder_hidden')
        dense = linear(inputs)
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
    def decode(latent_code):
        """
        Builds the decoder part of the VAE
        """

        # Create regular hidden layers
        linear = snt.Linear(self.num_units,
                            name='decoder_hidden')
        dense = linear(latent_code)
        dense = tf.nn.relu(dense)

        dense = linear(dense)
        dense = tf.nn.relu(dense)

        output = linear(dense)
        output = tf.reshape(output, [-1] + self.input_shape)

        return output


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
        self.latent_prior = Normal(
            loc=np.zeros(self.latent_dim, dtype=np.float32),
            scale=np.ones(self.latent_dim, dtype=np.float32))

        # Define the encoding - decoding process
        q_mu, q_sigma = self.encode(inputs)

        self.q_distribution = Normal(loc=q_mu, scale=q_sigma)
        assert self.q_distribution.reparameterization_type == FULLY_REPARAMETERIZED

        z = self.q_distribution.sample()

        outputs = decoder(z)
        decoder_bern = Bernoulli(logits=outputs)

        return decoder_bern.sample()


