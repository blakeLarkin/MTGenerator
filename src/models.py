from __future__ import division, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.recurrent import lstm
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation

from utils import *

def artToPrimaryTypeModel(numCategories, checkpoint_path='./classifier_checkpoints/',
                            best_checkpoint_path='./best_classifier_checkpoints/'):
  '''
  Convolutional network model for categorizing card art into primary types
  Inputs:
    numCategories: total number of valid categories
    checkpoint_path: path to save model after every epoch ('./classifier_checkpoints/')
    best_checkpoint_path: path to save the best models by
      validation ('./best_classifier_checkpoints/')
  Outputs:
    model: convolutional model, ready to be trained
  '''
  # Data Preprocessing and Augmentation
  preprocessor = ImagePreprocessing()
  preprocessor.add_featurewise_zero_center(mean=102.59733902)
  preprocessor.add_featurewise_stdnorm(std=65.9076363382)

  augmentor = ImageAugmentation()
  augmentor.add_random_flip_leftright()

  # Model Structure
  network = input_data(shape=[None, 64, 64, 3],
    data_preprocessing=preprocessor,
    data_augmentation=augmentor)
  network = conv_2d(network, 64, 4, activation='relu')
  network = max_pool_2d(network, 2)
  network = conv_2d(network, 128, 3, activation='relu')
  network = conv_2d(network, 256, 3, activation='relu')
  network = max_pool_2d(network, 2)
  network = fully_connected(network, 1024, activation='relu')
  network = fully_connected(network, 512, activation='relu')
  network = dropout(network, 0.5)
  network = fully_connected(network, numCategories, activation='softmax')
  network = regression(network, optimizer='adam', loss='categorical_crossentropy',
                        learning_rate=0.001)

  model = tflearn.DNN(network, tensorboard_verbose=0, checkpoint_path='./classifier_checkpoints/',
                        best_checkpoint_path='./best_classifier_checkpoints')
  return model

def typeSubtypeNameGeneratorModel(maxLength, charIndex, checkpoint_path='./generator_checkpoints/'):
  '''
  Recurrent network model for generating card types, subtypes, and names
  Inputs:
    maxLength: the maximum length for a generated sequence
    charIndex: map from chars to the index they represent in a onehot encoding
    checkpoint_path: path to save model after every epoch ('./generator_checkpoints/')
  '''
  network = input_data(shape=[None, maxLength, len(charIndex)])
  network = lstm(network, 512, return_seq=True)
  network = dropout(network, 0.5)
  network = lstm(network, 512, return_seq=True)
  network = dropout(network, 0.5)
  network = lstm(network, 512)
  network = dropout(network, 0.5)
  network = fully_connected(network, len(charIndex), activation='softmax')
  network = regression(network, optimizer='adam', loss='categorical_crossentropy',
                        learning_rate=0.001)

  model = tflearn.SequenceGenerator(network, tensorboard_verbose=0, dictionary=charIndex,
                                      seq_maxlen=maxLength, clip_gradients=5.0,
                                      checkpoint_path='./generator_checkpoints/')
  return model
