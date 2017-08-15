from __future__ import division, absolute_import

import tflearn
from tflearn.data_utils import shuffle, to_categorical

from utils import *
from models import *

def trainArtToPrimaryTypeModel(artPath, jsonPath, testProp, numEpochs=50):
  '''
  Trains a convolutional network to categorize card art by primary type
  Inputs:
    artPath: path to card art
    jsonPath: path to card data json file
    testProp: proportion of samples to be used for test/validation
    numEpochs: number of epochs to train for (50)
  '''
  (X, Y), (X_Test, Y_Test), numCategories = turnPicsToSimpleInputs(artPath,
                                                                    jsonPath,
                                                                    testProp=testProp)
  X, Y = shuffle(X, Y)
  Y = to_categorical(Y, numCategories)
  Y_Test = to_categorical(Y_Test, numCategories)

  # Train model as classifier
  model = artToMainTypeModel(numCategories)
  model.fit(X, Y, n_epoch=numEpochs, shuffle=True, validation_set=(X_Test, Y_Test),
              show_metric=True, batch_size=100, run_id='mtg_classifier')

def trainTypeSubtypeNameGenerator(jsonPath, testProp, maxLength, numEpochs=50):
  '''
  Trains a recurrent network to generate card types, subtypes, and names
  Inputs:
    jsonPath: path to card data json file
    testProp: proportion of samples to be used for test/validation
    maxLength: the maximum length of a single generated sample
    numEpochs: number of epochs to train for (50)
  '''
  (X, Y, charIndex), totalString = simpleGenerateTypeSubtypeToNameInputs(jsonPath, maxLength)

  model = typeSubtypeNameGeneratorModel(maxLength, charIndex)
  
  for i in range(numEpochs):
    randomIndex = random.randint(0, len(totalString) -  maxLength - 1)
    seed = totalString[randomIndex:randomIndex + maxLength]
    model.fit(X, Y, validation_set=testProp, batch_size=128, n_epoch=1, show_metric=True,
                run_id='typeSubtypeName')
    print("-- TESTING...")
    print("-- Test with temperature of 1.0 --")
    print(model.generate(600, temperature=1.0, seq_seed=seed))
    print("-- Test with temperature of 0.5 --")
    print(model.generate(600, temperature=0.5, seq_seed=seed))
