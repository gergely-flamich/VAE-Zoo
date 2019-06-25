import tensorflow as tf

import argparse
import os, tempfile


def is_valid_file(parser, arg):
    """
    Taken from
    https://stackoverflow.com/questions/11540854/file-as-command-line-argument-for-argparse-error-message-if-argument-is-not-va
    and
    https://stackoverflow.com/questions/9532499/check-whether-a-path-is-valid-in-python-without-creating-a-file-at-the-paths-ta
    """
    arg = str(arg)
    if os.path.exists(arg):
        return arg

    dirname = os.path.dirname(arg) or os.getcwd()
    try:
        with tempfile.TemporaryFile(dir=dirname): pass
        return arg
    except Exception:
        parser.error("A file at the given path cannot be created: " % arg)

# ==============================================================================
# Tensorflow helper functions
# ==============================================================================

def setup_eager_checkpoints_and_restore(variables, checkpoint_dir, checkpoint_name="_ckpt"):
    """
    Convenience function to set up TF eager checkpoints for the given variables.

    variables - iterable of TF Variables that we want to checkpoint
    checkpoint_dir - string containing the checkpoint path
    """

    ckpt_prefix = os.path.join(checkpoint_dir, checkpoint_name)

    checkpoint = tf.train.Checkpoint(**{v.name: v for v in variables})

    latest_checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)

    if latest_checkpoint_path is None:
        print("No checkpoint found!")
    else:
        print("Checkpoint found at {}, restoring...".format(latest_checkpoint_path))
        checkpoint.restore(latest_checkpoint_path).assert_consumed()
        print("Model restored!")

    return checkpoint, ckpt_prefix


