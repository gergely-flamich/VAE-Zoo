import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

LEARNING_RATE = 1e-3

def encoder(x):

    num_inputs = 28*28
    num_units = 300
    num_z = 2

    input_shape = [-1, 28 * 28]

    inputs = tf.reshape(x, input_shape)

    # The encoder is q_theta(z | x)
    # We assume that each z_i is Gaussian with mean mu and variance sigma

    dense1 = tf.layers.dense(
        inputs=inputs,
        units=num_units,
        activation=tf.nn.relu
    )

    mu = tf.layers.dense(
        inputs=dense1,
        units=num_z,
        activation=None
    )

    sigma = tf.layers.dense(
        inputs=dense1,
        units=num_z,
        activation=tf.nn.softplus
    )

    return mu, sigma

def decoder(z):

    num_inputs = 28*28
    num_units = 300
    num_z = 2

    dense2 = tf.layers.dense(
        inputs=z,
        units=num_units,
        activation=tf.nn.relu
    )

    dense3 = tf.layers.dense(
        inputs=dense2,
        units=num_units,
        activation=tf.nn.relu
    )


    outputs = tf.layers.dense(
        inputs=dense3,
        units=num_inputs,
        activation=None
    )

    outputs = tf.reshape(outputs, [-1, 28, 28])

    return outputs

def create_model(features, params):

    num_inputs = 28*28
    num_units = 300
    num_z = 2

    # ==========================================================================
    # Encoder
    # ==========================================================================



    # W_1 = tf.get_variable("encoder_w_1", [num_inputs, num_units])
    # b_1 = tf.get_variable("encoder_b_1", [num_units])

    # encoder_L_1 = tf.matmul(inputs, W_1) + b_1
    # encoder_L_1 = tf.nn.relu(encoder_L_1)

    # W_2_mu = tf.get_variable("encoder_w_2_mu", [num_units, num_z])
    # b_2_mu = tf.get_variable("encoder_b_2_mu", [num_z])

    # encoder_mu = tf.matmul(encoder_L_1, W_2_mu) + b_2_mu

    # W_2_sigma = tf.get_variable("encoder_w_2_sigma", [num_units, num_z])
    # b_2_sigma = tf.get_variable("encoder_b_2_sigma", [num_z])

    # encoder_sigma = tf.matmul(encoder_L_1, W_2_sigma) + b_2_sigma
    # encoder_sigma = tf.nn.softplus(encoder_sigma)

    # ==========================================================================
    # Decoder
    # ==========================================================================

    # The decoder is p_phi(x | z)

    #epsilon = tf.distributions.Normal(loc=0., scale=1.).sample([num_z])

    #z = encoder_mu + encoder_sigma * epsilon

    q_mu, q_sigma = encoder(features)

    q_distribution = tfp.distributions.Normal(loc=q_mu, scale=q_sigma)

    assert q_distribution.reparameterization_type == tfp.distributions.FULLY_REPARAMETERIZED

    z = q_distribution.sample()

    outputs = decoder(z)

    # W_3 = tf.get_variable("decoder_w_3", [num_z, num_units])
    # b_3 = tf.get_variable("decoder_b_3", [num_units])

    # decoder_L_1 = tf.matmul(z, W_3) + b_3
    # decoder_L_1 = tf.nn.relu(decoder_L_1)

    # W_4 = tf.get_variable("decoder_w_4", [num_units, num_inputs])
    # b_4 = tf.get_variable("decoder_b_4", [num_inputs])

    # decoder_out = tf.matmul(decoder_L_1, W_4) + b_4
    # decoder_out = tf.nn.sigmoid(decoder_out)

    decoder_bern = tfp.distributions.Bernoulli(logits=outputs)

    post_pred_samp = decoder_bern.sample()

    tf.summary.image('posterior_predictive',
                     tf.cast(tf.reshape(post_pred_samp, [-1, 28, 28, 1]), tf.float32))

    # log prior on hidden units
    # log_prior = -num_z/2.0 * np.log(2 * np.pi) - 0.5 * tf.norm(z)

    # # log variational posterior
    # log_var_post = -num_z/2.0 * (np.log(2 * np.pi) + 2 * tf.reduce_sum(tf.log(encoder_sigma)))
    # log_var_post += - 0.5 * tf.norm(tf.divide(z - encoder_mu, encoder_sigma))

    hidden_prior = tfp.distributions.Normal(loc=np.zeros(num_z, dtype=np.float32),
                                           scale=np.ones(num_z, dtype=np.float32))
    # kl_divergence = log_var_post - log_prior

    kl_divergence = tf.reduce_sum(tfp.distributions.kl_divergence(q_distribution, hidden_prior), 1)

    loss = tf.reduce_sum(decoder_bern.log_prob(features), [1, 2]) # labels * tf.log(decoder_out) + (1 - labels) * tf.log(1 - decoder_out)

    loss = tf.reduce_sum(kl_divergence - loss, 0)

    return outputs, loss


def baseline_vae_model_fn(features, labels, mode, params):

    decoder_out, loss = create_model(features, params)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=decoder_out)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE)
        train_op = optimizer.minimize(loss=loss,
                                      global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(labels=labels,
                                            predictions=decoder_out)
        }

        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          eval_metric_ops=eval_metric_ops)


def baseline_vae_input_fn():

    ((train_data, train_labels),
     (eval_data, eval_labels)) = tf.keras.datasets.mnist.load_data()

    dataset = tf.data.Dataset.from_tensor_slices(train_data)
    dataset = dataset.shuffle(100000)
    dataset = dataset.repeat(5)
    dataset = dataset.map(map_func=to_black_and_white)
    dataset = dataset.batch(batch_size=100)

    return dataset

def to_black_and_white(img):

    return tf.cast((img < 128), tf.float32)

