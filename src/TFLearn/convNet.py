from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.recurrent import lstm
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation

from utils import *

def artToMainTypeModel(artPath, jsonPath, testProp):
  (X, Y), (X_Test, Y_Test), numCategories = turnPicsToSimpleInputs(artPath, jsonPath, testProp=testProp)
  X, Y = shuffle(X, Y)
  Y = to_categorical(Y, numCategories + 1)
  print('test', Y[0])
  Y_Test = to_categorical(Y_Test, numCategories + 1)

  # Data Preprocessing 
  preprocessor = ImagePreprocessing()
  preprocessor.add_featurewise_zero_center()
  preprocessor.add_featurewise_stdnorm()

  # Data Augmenter
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
  network = fully_connected(network, numCategories + 1, activation='softmax')
  network = regression(network, optimizer='adam', loss='categorical_crossentropy', learning_rate=0.001)

  # Train model as classifier
  model = tflearn.DNN(network, tensorboard_verbose=0, checkpoint_path='./classifier_checkpoints/', best_checkpoint_path='./best_classifier_checkpoints')
  model.fit(X, Y, n_epoch=50, shuffle=True, validation_set=(X_Test, Y_Test), show_metric=True, batch_size=100, run_id='mtg_classifier')

def typeSubtypeNameGenerator(jsonPath, testProp, maxLength):
  # TODO: check why maxLength is messing up
  (X, Y, charIndex), totalString = simpleGenerateTypeSubtypeToNameInputs(jsonPath, maxLength)

  print(X)
  print(len(X))
  print(len(X[0]))
  print(len(X[0][0]))

  network = input_data(shape=[None, maxLength, len(charIndex)])
  network = lstm(network, 512, return_seq=True)
  network = dropout(network, 0.5)
  network = lstm(network, 512, return_seq=True)
  network = dropout(network, 0.5)
  network = lstm(network, 512)
  network = dropout(network, 0.5)
  network = fully_connected(network, len(charIndex), activation='softmax')
  network = regression(network, optimizer='adam', loss='categorical_crossentropy', learning_rate=0.001)

  # Train model as a generator
  model = tflearn.SequenceGenerator(network, tensorboard_verbose=0, dictionary=charIndex, seq_maxlen=maxLength, clip_gradients=5.0, checkpoint_path='./generator_checkpoints/')
  for i in range(50):
    randomIndex = random.randint(0, len(totalString) -  maxLength - 1)
    seed = totalString[randomIndex:randomIndex + maxLength]
    model.fit(X, Y, validation_set=testProp, batch_size=128, n_epoch=1, show_metric=True, run_id='typeSubtypeName')
    print("-- TESTING...")
    print("-- Test with temperature of 1.0 --")
    print(m.generate(600, temperature=1.0, seq_seed=seed))
    print("-- Test with temperature of 0.5 --")
    print(m.generate(600, temperature=0.5, seq_seed=seed))
