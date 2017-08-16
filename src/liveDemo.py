from __future__ import division, absolute_import

import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.recurrent import lstm
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import io
import json

from utils import *
from models import *

def demoArtToPrimaryTypeNetwork(artPath, cardPath, jsonPath, modelPath, numDesired=10,
                                  showPics=False):
  '''
  Loads and tests a trained convolutional classifier model for a live demo
  Inputs:
    artPath: path to card art
    cardPath: path to card scans
    jsonPath: path to card data json file
    modelPath: path to trained model
    numDesired: size of subset for demo (10)
    showPics: boolean for wether or not card art/scans should be displayed (False)
  '''
  inputNames, inputs, numCategories, categoryToType, cardNameToCategories = \
    getLiveDemoPicsToInput(artPath, cardPath, jsonPath, numDesired=numDesired, showPics=showPics)

  model = artToPrimaryTypeModel(numCategories)
  model.load(modelPath, weights_only=True)

  jsonFile = io.open(jsonPath)
  cardData = json.load(jsonFile)

  numCorrect = 0
  wrongCards = []

  for i in range(numDesired):
    print('\nCard Name: ' + inputNames[i])
    prediction = (model.predict([inputs[i]])).tolist()[0]
    category = 0
    for j in range(len(prediction)):
      if prediction[j] > prediction[category]:
        category = j
    print('Prediction: ' + categoryToType[category])
    print('Actual: ' + categoryToType[cardNameToCategories[inputNames[i]]] + '\n')
    if categoryToType[category] == categoryToType[cardNameToCategories[inputNames[i]]]:
      numCorrect+=1
    else:
      wrongCards.append(inputNames[i])
  print('Percentage Correct: %2.2f%%' % (numCorrect * 100 / numDesired))
  print('Wrong Cards: ' + str(wrongCards))

def demo1():
  '''
  Runs first live demo, 3 primary type classifications with art showing
  '''
  demoArtToPrimaryTypeNetwork('../data/mtg cards/art/', '../data/mtg cards/cards/',
                                '../data/mtg data/AllCards.json',
                                './best_classifier_checkpoints9533', numDesired=3, showPics=True)

def demo2():
  '''
  Runs second live demo, 50 primary type classification without art showing
  '''
  demoArtToPrimaryTypeNetwork('../data/mtg cards/art/', '../data/mtg cards/cards/',
                                '../data/mtg data/AllCards.json',
                                './best_classifier_checkpoints9533', numDesired=1000)

