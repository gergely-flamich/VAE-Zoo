import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfk = tf.keras
tfl = tf.keras.layers
tfs = tf.summary

class ManifoldEncoder(tfl.Layer):
    
    def __init__(self,
                 latent_dim=64,
                 name="manifold_encoder",
                 **kwargs):
        
        # Initialise superclass
        super(ManifoldEncoder, self).__init__(name=name, **kwargs)
        
        # Set fields
        self.latent_dim = latent_dim
        
        # --------------------------------------------------------------
        # Define layers
        # --------------------------------------------------------------
        
        self.layers = [
            tfl.Reshape((128, 128, 3), input_shape=(128, 128, 3)),
            tfl.Conv2D(filters=64,
                       kernel_size=(5, 5),
                       strides=2,
                       padding="same",
                       name="encoder_conv1"),
            tf.nn.leaky_relu,
            tfl.Conv2D(filters=128,
                       kernel_size=(5, 5),
                       strides=2,
                       padding="same",
                       use_bias=False,
                       name="encoder_conv2"),
            tfl.BatchNormalization(),
            tf.nn.leaky_relu,
            tfl.Conv2D(filters=128,
                       kernel_size=(5, 5),
                       strides=2,
                       padding="same",
                       use_bias=False,
                       name="encoder_conv3"),
            tfl.BatchNormalization(),
            tf.nn.leaky_relu,
            tfl.Conv2D(filters=128,
                       kernel_size=(5, 5),
                       strides=2,
                       padding="same",
                       use_bias=False,
                       name="encoder_conv3"),
            tfl.BatchNormalization(),
            tf.nn.leaky_relu,
            tfl.Flatten(),
            tfl.Dense(units=1024,
                      use_bias=False,
                      name="encoder_linear1"),
            tfl.BatchNormalization(),
            tf.nn.leaky_relu
        ]
        
        self.loc_head = tfl.Dense(units=self.latent_dim,
                                  name="encoder_loc_head")
        self.log_scale_head = tfl.Dense(units=self.latent_dim,
                                        name="encoder_scale_head")
        
    def call(self, inputs, training=True):
        
        activations = inputs
        
        for layer in self.layers:
            
            if isinstance(layer, tfl.BatchNormalization):
                activations = layer(activations, training=training)
                
            else:
                activations = layer(activations)
            
        loc = self.loc_head(activations)
        scale = tf.nn.softplus(self.log_scale_head(activations))
        
        self.posterior = tfd.Normal(loc=loc, scale=scale)
        
        return self.posterior.sample()
    
    
class ManifoldDecoder(tfl.Layer):
    
    def __init__(self, 
                 name="manifold_decoder",
                 **kwargs):
        
        # Initialise superclass
        super(ManifoldDecoder, self).__init__(name=name, **kwargs)
        
        # Hyperparams
        self.log_gamma = tf.Variable(0., "log_gamma")
        
        # --------------------------------------------------------------
        # Define layers: reverse of a ManifoldEncoder
        # --------------------------------------------------------------
        self.layers = [
            tfl.Dense(units=1024),
            tf.nn.leaky_relu,
            tfl.Dense(units=8 * 8 * 128,
                      use_bias=False),
            tfl.BatchNormalization(),
            tf.nn.leaky_relu,
            tfl.Reshape((8, 8, 128)),
            tfl.Conv2DTranspose(filters=128,
                                kernel_size=(5, 5),
                                strides=2,
                                padding="same",
                                use_bias=False),
            tfl.BatchNormalization(),
            tf.nn.leaky_relu,
            tfl.Conv2DTranspose(filters=128,
                                kernel_size=(5, 5),
                                strides=2,
                                padding="same",
                                use_bias=False),
            tfl.BatchNormalization(),
            tf.nn.leaky_relu,
            tfl.Conv2DTranspose(filters=64,
                                kernel_size=(5, 5),
                                strides=2,
                                padding="same",
                                use_bias=False),
            tfl.BatchNormalization(),
            tf.nn.leaky_relu,
            tfl.Conv2DTranspose(filters=3,
                                kernel_size=(5, 5),
                                padding="same",
                                strides=2)
        ]
        
    def call(self, inputs, training=True):
        
        activations = inputs
        
        for layer in self.layers:
            if isinstance(layer, tfl.BatchNormalization):
                activations = layer(activations, training=training)
                
            else:
                activations = layer(activations)
            
        activations = tf.nn.sigmoid(activations)
            
        likelihood_scale = tf.exp(self.log_gamma)
        self.likelihood = tfd.Normal(loc=activations,
                                     scale=likelihood_scale)
        
        return activations
    

class ManifoldVAE(tfk.Model):
    
    def __init__(self,
                 latent_dim=64,
                 name="manifold_vae",
                 **kwargs):
        
        super(ManifoldVAE, self).__init__(name=name,
                                          **kwargs)
        
        # Define stuff
        self.latent_dim = latent_dim
        self.prior = tfd.Normal(loc=tf.zeros(latent_dim),
                                scale=tf.ones(latent_dim))
        
        # Define blocks
        self.encoder = ManifoldEncoder(latent_dim=self.latent_dim)
        self.decoder = ManifoldDecoder()
        
    @property 
    def posterior(self):
        return self.encoder.posterior
    
    @property
    def likelihood(self):
        return self.decoder.likelihood
    
    @property
    def log_gamma(self):
        return self.decoder.log_gamma
    
    @property
    def kl_divergence(self):
        return tfd.kl_divergence(self.posterior, self.prior)
    
    @property
    def log_prob(self):
        return tf.reduce_sum(self._log_prob)
         
    def call(self, inputs, training=True):
        
        latents = self.encoder(inputs, training=training)
        reconstruction = self.decoder(latents, training=training)
        
        self._log_prob = self.likelihood.log_prob(inputs)
        
        return reconstruction
    
    
# ============================================================================
# ============================================================================
# Second VAE
# ============================================================================
# ============================================================================

class MeasureEncoder(tfl.Layer):
    
    def __init__(self,
                 second_stage_depth=4,
                 second_stage_units=1024,
                 latent_dim=64,
                 name="measure_encoder",
                 **kwargs):
        
        # Initialise superclass
        super(MeasureEncoder, self).__init__(name=name, **kwargs)
        
        # Set fields
        self.latent_dim = latent_dim
        self.second_stage_depth = second_stage_depth
        self.second_stage_units = second_stage_units
        
        # --------------------------------------------------------------
        # Define layers
        # --------------------------------------------------------------
        
        self.layers = [tfl.Dense(units=self.second_stage_units) for i in range(self.second_stage_depth)]
        
        self.loc_head = tfl.Dense(units=self.latent_dim,
                                  name="encoder_loc_head")
        self.log_scale_head = tfl.Dense(units=self.latent_dim,
                                        name="encoder_scale_head")
        
    def call(self, inputs):
        
        activations = inputs
        
        for layer in self.layers:
            
            activations = tf.nn.relu(layer(activations))
            
        # Add residual connection
        activations = tf.concat([activations, inputs], axis=-1)
            
        loc = self.loc_head(activations)
        scale = tf.exp(self.log_scale_head(activations))
        
        self.posterior = tfd.Normal(loc=loc, scale=scale)
        
        return self.posterior.sample()
    
    
class MeasureDecoder(tfl.Layer):
    
    def __init__(self, 
                 latent_dim=64,
                 second_stage_units=1024,
                 second_stage_depth=4,
                 name="measure_decoder",
                 **kwargs):
        
        # Initialise superclass
        super(MeasureDecoder, self).__init__(name=name, **kwargs)
        
        self.latent_dim = latent_dim
        self.second_stage_depth = second_stage_depth
        self.second_stage_units = second_stage_units
        
        # Hyperparams
        self.log_gamma = tf.Variable(0., "log_gamma")
        
        # --------------------------------------------------------------
        # Define layers: reverse of a MeasureEncoder
        # --------------------------------------------------------------
        self.layers = [tfl.Dense(units=self.second_stage_units) for i in range(self.second_stage_depth - 1)]
        self.latent_head = tfl.Dense(units=self.latent_dim)
        
        
    def call(self, inputs):
        
        activations = inputs
        
        for layer in self.layers:
            
            activations = tf.nn.relu(layer(activations))
            
        # Add residual connection
        activations = tf.concat([activations, inputs], axis=-1)
        activations = self.latent_head(activations)
            
        likelihood_scale = tf.exp(self.log_gamma)
        self.likelihood = tfd.Normal(loc=activations,
                                     scale=likelihood_scale)
        
        return activations
    

class MeasureVAE(tfk.Model):
    
    def __init__(self,
                 latent_dim=64,
                 second_stage_depth=4,
                 name="measure_vae",
                 **kwargs):
        
        super(MeasureVAE, self).__init__(name=name,
                                          **kwargs)
        
        # Define stuff
        self.latent_dim = latent_dim
        self.prior = tfd.Normal(loc=tf.zeros(latent_dim),
                                scale=tf.ones(latent_dim))
        
        # Define blocks
        self.encoder = MeasureEncoder(latent_dim=self.latent_dim, 
                                      second_stage_depth=second_stage_depth)
        
        self.decoder = MeasureDecoder(second_stage_depth=second_stage_depth,
                                      latent_dim=self.latent_dim)
        
    @property 
    def posterior(self):
        return self.encoder.posterior
    
    @property
    def likelihood(self):
        return self.decoder.likelihood
    
    @property
    def log_gamma(self):
        return self.decoder.log_gamma
    
    @property
    def kl_divergence(self):
        return tfd.kl_divergence(self.posterior, self.prior)
    
    @property
    def log_prob(self):
        return tf.reduce_sum(self._log_prob)
         
    def call(self, inputs, training=True):
        
        latents = self.encoder(inputs)
        reconstruction = self.decoder(latents)
        
        self._log_prob = self.likelihood.log_prob(inputs)
        
        return reconstruction