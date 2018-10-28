"""
	This file defines all the parameters for the experiment.
	Author: Marcel Santos (mss8@cin.ufpe.br), Federal University of Pernambuco (UFPE).
"""

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_boolean ('saveModel', 1, 
                            'Indicate whether the network is training or testing.')

tf.app.flags.DEFINE_string  ('trainImageDir', 'dataset/train/input/',
                            'Directory for the train set input.')

tf.app.flags.DEFINE_string  ('trainLabelsDir', 'dataset/train/segments/',
                            'Directory for the train set labels.')

tf.app.flags.DEFINE_string  ('valImagesDir', 'dataset/val/input/',
                            'Directory for the validation set input.')

tf.app.flags.DEFINE_string  ('valLabelsDir', 'dataset/val/segments/',
                            'Directory for the validation set labels.')
