import os, json, argparse

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfk = tf.keras
tfl = tf.keras.layers
tfs = tf.summary

import tensorflow_datasets as tfds

import numpy as np

from tqdm import tqdm

from architectures import ManifoldVAE, MeasureVAE
from utils import is_valid_file


MODEL_DIR = "/tmp/2-stage-vae-v3/"

config = {
        "num_training_examples": 60000,
        "batch_size": 64,
        "num_epochs": 400,
        "num_epochs_stage_2": 800,
        
        "num_latents": 128,
    
        "beta1": 1.,
        "beta2": 1.,
        "warmup": 10.,
        
        "learning_rate": 1e-3,
        
        "optimizer": "adam",
        
        "checkpoint_name": "_ckpt",
        "log_freq": 100,
}

datasets = {
    
    "mnist": {
        "num_training_examples": 60000,
    },
    
    "celeba": {
        "num_training_examples": 162770,
    }
}

def parse_celeba(datum):
    return tf.cast(datum["image"][45:-45, 25:-25, :], tf.float32) / 255.

def load_celeba_data(train=True, data_dir="/tmp/tensorflow_datasets", batch_size=64, shuffle_buffer=2048):
    
    data, info = tfds.load("celeb_a", with_info=True, as_supervised=False, data_dir=data_dir)
    train_data, test_data = data['train'], data['test']
    
    if train:
        celeba = train_data
    else:
        celeba = test_data
    
    celeba = celeba.map(parse_celeba).shuffle(shuffle_buffer).batch(batch_size)

    celeba = celeba.prefetch(tf.data.experimental.AUTOTUNE)
    
    return celeba


def mnist_input_fn(data, batch_size=256, shuffle_samples=5000):
    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = dataset.shuffle(shuffle_samples)
    dataset = dataset.map(mnist_parse_fn)
    dataset = dataset.batch(batch_size)

    return dataset


def mnist_parse_fn(data):
    return tf.cast(data, tf.float32) / 255.


optimizers = {
    "sgd": tfk.optimizers.SGD,
    "adam": tfk.optimizers.Adam,
}

def run(model_dir,
        dataset_name,
        train_first=True,
        train_second=True):
    
    print("Using " + dataset_name)

    num_batches = datasets[dataset_name]["num_training_examples"] // config["batch_size"] + 1

    print("Configuration:")
    print(json.dumps(config, indent=4, sort_keys=True))

    # ==========================================================================
    # Load dataset
    # ==========================================================================

    if dataset_name == "mnist":
        ((train_data, _),
         (eval_data, _)) = tf.keras.datasets.mnist.load_data()

    # ==========================================================================
    # Create model
    # ==========================================================================

    manifold_vae = ManifoldVAE(latent_dim=config["num_latents"])
    measure_vae = MeasureVAE(latent_dim=config["num_latents"])

    manifold_optimizer = optimizers[config["optimizer"]](config["learning_rate"])
    measure_optimizer = optimizers[config["optimizer"]](config["learning_rate"])

    # ==========================================================================
    # Checkpoints
    # ==========================================================================

    # Create checkpoint and its manager
    manifold_ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=manifold_optimizer, net=manifold_vae)
    manifold_manager = tf.train.CheckpointManager(manifold_ckpt, model_dir + "/manifold_checkpoints", max_to_keep=3)

    measure_ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=measure_optimizer, net=measure_vae)
    measure_manager = tf.train.CheckpointManager(measure_ckpt, model_dir + "/measure_checkpoints", max_to_keep=3)

    # Attempt to restore model
    manifold_ckpt.restore(manifold_manager.latest_checkpoint)
    if manifold_manager.latest_checkpoint:
        print("Restored from {}".format(manifold_manager.latest_checkpoint))
    else:
        print("Initializing Manifold VAE from scratch.")

    # Attempt to restore model
    measure_ckpt.restore(measure_manager.latest_checkpoint)
    if measure_manager.latest_checkpoint:
        print("Restored from {}".format(measure_manager.latest_checkpoint))
    else:
        print("Initializing Measure VAE from scratch.")
    # ==========================================================================
    # Train the model
    # ==========================================================================

    def train_first_stage(log_freq=10, save_freq=50):

        beta = config["beta1"]

        for epoch in range(1, config["num_epochs"] + 1):

            if dataset_name == "mnist":
                dataset = mnist_input_fn(data=train_data,
                                        batch_size=config["batch_size"])
            elif dataset_name == "celeba":
                dataset = load_celeba_data(batch_size=config["batch_size"])

            with tqdm(total=num_batches) as pbar:
                for batch in dataset:

                    # Increment checkpoints step
                    manifold_ckpt.step.assign_add(1)

                    with tf.GradientTape() as tape:

                        output = manifold_vae(batch, training=True)

                        kl = manifold_vae.kl_divergence
                        total_kl = tf.reduce_sum(kl)

                        log_prob = manifold_vae.log_prob

                        warmup_coef = 1. #tf.minimum(1., manifold_optimizer.iterations.numpy() / (config["warmup"] * num_batches))

                        # negative ELBO
                        loss = total_kl - beta * warmup_coef * log_prob 

                        output = tf.cast(output, tf.float32)

                    gradients = tape.gradient(loss, manifold_vae.trainable_variables)
                    manifold_optimizer.apply_gradients(zip(gradients, manifold_vae.trainable_variables))

                    # Log stuff
                    if tf.equal(manifold_optimizer.iterations % log_freq, 0):
                        # Add tensorboard summaries
                        tfs.scalar("Loss", loss, step=manifold_optimizer.iterations)
                        tfs.scalar("Total_KL", total_kl, step=manifold_optimizer.iterations)
                        tfs.scalar("Max_KL", tf.reduce_max(kl), step=manifold_optimizer.iterations)
                        tfs.scalar("Log-Probability", log_prob, step=manifold_optimizer.iterations)
                        tfs.scalar("Warmup_Coef", warmup_coef, step=manifold_optimizer.iterations)
                        tfs.scalar("Gamma-x", tf.exp(manifold_vae.log_gamma), step=manifold_optimizer.iterations)
                        tfs.image("Reconstruction", output, step=manifold_optimizer.iterations)

                    if tf.equal(manifold_ckpt.step % save_freq, 0):
                        manifold_manager.save()

                    # Update the progress bar
                    pbar.update(1)
                    pbar.set_description("Epoch {}, ELBO: {:.2f}".format(epoch, loss))

        print("First Stage Training Complete!")

    def train_second_stage(log_freq=10, save_freq=50):

        beta = config["beta2"]

        for epoch in range(1, config["num_epochs_stage_2"] + 1):

            if dataset_name == "mnist":
                dataset = mnist_input_fn(data=train_data,
                                        batch_size=config["batch_size"])
            elif dataset_name == "celeba":
                dataset = load_celeba_data(batch_size=config["batch_size"])

            with tqdm(total=num_batches) as pbar:
                for batch in dataset:

                    # Increment checkpoints step
                    measure_ckpt.step.assign_add(1)

                    with tf.GradientTape() as tape:

                        latents = manifold_vae.encoder(batch, training=False)

                        output = measure_vae(latents)

                        kl = measure_vae.kl_divergence
                        total_kl = tf.reduce_sum(kl)

                        log_prob = measure_vae.log_prob

                        warmup_coef = 1. #tf.minimum(1., measure_optimizer.iterations.numpy() / (config["warmup"] * num_batches))

                        # negative ELBO
                        loss = total_kl - beta * warmup_coef * log_prob 

                        output = tf.cast(output, tf.float32)

                    gradients = tape.gradient(loss, measure_vae.trainable_variables)
                    measure_optimizer.apply_gradients(zip(gradients, measure_vae.trainable_variables))

                    # Log stuff
                    if tf.equal(measure_optimizer.iterations % log_freq, 0):
                        # Add tensorboard summaries
                        tfs.scalar("2-Loss", loss, step=measure_optimizer.iterations)
                        tfs.scalar("2-Total_KL", total_kl, step=measure_optimizer.iterations)
                        tfs.scalar("2-Max_KL", tf.reduce_max(kl), step=measure_optimizer.iterations)
                        tfs.scalar("2-Log-Probability", log_prob, step=measure_optimizer.iterations)
                        tfs.scalar("2-Warmup_Coef", warmup_coef, step=measure_optimizer.iterations)
                        tfs.scalar("2-Gamma-z", tf.exp(measure_vae.log_gamma), step=measure_optimizer.iterations)

                    if tf.equal(measure_ckpt.step % save_freq, 0):
                        measure_manager.save()

                    # Update the progress bar
                    pbar.update(1)
                    pbar.set_description("Epoch {}, ELBO: {:.2f}".format(epoch, loss))

        print("Second Stage Training Complete!")

    train_summary_writer = tfs.create_file_writer(model_dir + '/summaries/train')            

    with train_summary_writer.as_default():
        train_first_stage(log_freq=50)

        manifold_vae.trainable = False

        train_second_stage(log_freq=50)
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Experimental models for 2-stage VAEs')

    parser.add_argument('--config', type=open, default=None,
                    help='Path to the config JSON file.')

    parser.add_argument('--dataset',
                        choices=list(datasets),
                        default="mnist")
    
    parser.add_argument('--no_train_first',
                        action="store_false",
                        dest="train_first",
                        default=True)
    
    parser.add_argument('--no_train_second',
                        action="store_false",
                        dest="train_second",
                        default=True)
    
    parser.add_argument('--model_dir',
                        type=lambda x: is_valid_file(parser, x),
                        default=MODEL_DIR,
                        help='The model directory.')

    args = parser.parse_args()
    
    run(train_first=args.train_first,
        train_second=args.train_second,
        dataset_name=args.dataset,
        model_dir=args.model_dir)
