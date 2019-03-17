import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
import numpy as np
import argparse
import matplotlib.pyplot as plt

from vae import baseline_vae_model_fn, baseline_vae_input_fn

models = {
    "baseline": baseline_vae_model_fn
}

def run(args):

    #print(train_data.shape)
    #print(train_data[0])

    classifier = tf.estimator.Estimator(model_fn=baseline_vae_model_fn,
                                        model_dir="/tmp/vae_mnist_baseline_model",
                                        params={
                                        })

    # train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": train_data},
    #                                                     y=train_data,
    #                                                     batch_size=100,
    #                                                     num_epochs=None,
    #                                                     shuffle=True)

    print("Beggining training of the VAE!")
    classifier.train(input_fn=baseline_vae_input_fn)

    print("Training finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VAEs')

    args = parser.parse_args()

    run(args)
