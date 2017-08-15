from __future__ import division, absolute_import

import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.recurrent import lstm
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from numpy import nonzero

from utils import *
from models import *

def demoArtToPrimaryTypeNetwork(artPath, cardPath, jsonPath, modelPath, numDesired=10):
  inputNames, inputs, numCategories, categoryToType, cardNameToCategories = getLiveDemoPicsToInput(artPath, cardPath, jsonPath, numDesired=numDesired)

  print('categoryToType')
  print(categoryToType)
  print(numCategories)

  model = artToPrimaryTypeModel(numCategories)
  model.load(modelPath, weights_only=True)

  jsonFile = io.open(jsonPath)
  cardData = json.load(jsonFile)

  print(categoryToType)
  for i in range(numDesired):
    print('\nCard Name: ' + inputNames[i])
    prediction = (model.predict([inputs[i]])).tolist()[0]
    category = 0
    for j in range(len(prediction)):
      if prediction[j] > prediction[category]:
        category = j
    print('Prediction: ' + categoryToType[category])
    print('Actual: ' + categoryToType[cardNameToCategories[inputNames[i]]] + '\n')

