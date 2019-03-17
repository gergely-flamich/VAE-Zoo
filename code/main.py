import tensorflow as tf
import argparse
import matplotlib.pyplot as plt

from tqdm import tqdm

from utils import is_valid_file
from vae import VAE


tf.enable_eager_execution()


def mnist_input_fn(data, batch_size=256, shuffle_samples=5000):
    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = dataset.shuffle(shuffle_samples)
    dataset = dataset.map(mnist_parse_fn)
    dataset = dataset.batch(batch_size)

    return dataset


def mnist_parse_fn(data):
    return tf.cast(data, tf.float32) / 255.


def run(args):

    # ==========================================================================
    # Configuration
    # ==========================================================================

    config = {
        "num_training_examples": 60000,
        "batch_size": 250,
        "num_epochs": 10,
        "learning_rate": 1e-3,
    }

    num_batches = config["num_training_examples"] // config["batch_size"] + 1

    # ==========================================================================
    # Load dataset
    # ==========================================================================

    ((train_data, _),
     (eval_data, _)) = tf.keras.datasets.mnist.load_data()

    # ==========================================================================
    # Create model
    # ==========================================================================

    vae = VAE(input_shape=(28, 28),
              num_units=400)

    optimizer = tf.train.RMSPropOptimizer(learning_rate=config["learning_rate"])

    # ==========================================================================
    # Train the model
    # ==========================================================================

    for epoch in range(1, config["num_epochs"] + 1):

        dataset = mnist_input_fn(data=train_data,
                                 batch_size=config["batch_size"])

        with tqdm(total=num_batches) as pbar:
            for batch in dataset:

                # Record gradients of the forward pass
                with tf.GradientTape() as tape:

                    output = vae(batch)

                    # negative ELBO
                    loss = vae.kl_divergence() - vae.input_log_prob()

                # Backprop
                grads = tape.gradient(loss, vae.get_all_variables())
                optimizer.apply_gradients(zip(grads, vae.get_all_variables()))

                # Update the progress bar
                pbar.update(1)
                pbar.set_description("Epoch {}, ELBO: {:.2f}".format(epoch, loss))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VAEs')

    parser.add_argument('--no_training', action="store_false", dest="is_training", default=True,
                    help='Should we just evaluate?')

    parser.add_argument('--model_dir', type=lambda x: is_valid_file(parser, x), default='/tmp/vae',
                    help='The model directory.')

    args = parser.parse_args()

    run(args)


