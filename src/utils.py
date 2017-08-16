from os import listdir
from PIL import Image
import random
import io
import json
import numpy as np

from tflearn.data_utils import string_to_semi_redundant_sequences

def generatePics(cardsPath, artPath='art', image_width=64, image_height=None, proportion=1):
  '''
  Creates images of the artwork section of magic the gathering cards
  Inputs:
    cardsPath: path to magic the gather card scans
    artPath: subpath that will be appended to cardsPath to store card artwork ('art')
    image_width: width in pixels of final artwork (64)
    image_height: height in pixel of final artwork, if None will be set to image_width (None)
    proportion: proportion of cards to create artwork for, for testing (1)
  '''
  if not image_height:
    image_height = image_width

  cards = listdir(cardsPath)
  print('Generating Card Art')
  for index, cardPath in enumerate(cards):
    if cardPath == '.DS_Store' or not random.random() < proportion :
      continue
    card = Image.open(cardsPath + cardPath)
    width, height = card.size
    artWidth, artHeight = (width * 0.76, height * 0.42)
    widthBorder = (width - artWidth) / 2
    heightBorder = widthBorder * 1.5
    art = card.crop((widthBorder, heightBorder, widthBorder + artWidth, heightBorder + artHeight))
    squareBorder = (artWidth - artHeight) / 2
    art = art.crop((squareBorder, 0, squareBorder + artHeight, artHeight))
    art = art.resize((image_width, image_height))
    art.save(cardsPath + '../' + artPath + '/' + cardPath)
    if int((index / len(cards)) * 10000) % 10 == 0:
      print ("Percent done: %2.1f %%" % ((index / len(cards)) * 100))
  print('Done')

def generateCardToSimpleTypeDict(jsonPath, cutoffSize=100):
  '''
  Creates a dictionary of card names to the card's primary type from a json file, only including
    primary types with a large enough representation. Creates a onehot relationship
  Inputs:
    jsonPath: path to magic the gather json file for card information
    cutoffSize: minimum size for a type to be included (100)
  Outputs:
    cardNameToCategoreis: dictionary from card names to the appropriate category of types
    numCategories: the total number of represented categories, not including the "other" category
  '''
  jsonFile = io.open(jsonPath)
  cardData = json.load(jsonFile)

  numOfType = {}

  for cardName in cardData.keys():
    if 'types' in cardData[cardName]:
      if not cardData[cardName]['types'][0] in numOfType:
        numOfType[cardData[cardName]['types'][0]] = 0
      numOfType[cardData[cardName]['types'][0]]+=1

  typeToCategory = {}
  numCategories = 0

  for type in numOfType:
    if numOfType[type] > cutoffSize:
      typeToCategory[type] = numCategories
      numCategories+=1
  typeToCategory['Other'] = numCategories
  numCategories+=1

  cardNameToCategories = {}

  for cardName in cardData.keys():
    category = typeToCategory['Other']
    if 'types' in cardData[cardName]:
      if cardData[cardName]['types'][0] in typeToCategory:
        category = typeToCategory[cardData[cardName]['types'][0]]
    cardNameToCategories[cardName] = category

  return (cardNameToCategories, numCategories, typeToCategory)

def representsInt(s):
  '''
  Returns True if the passed in string represents an int, False otherwise
  Inputs:
    s: string to check
  Outputs:
    True or False depending on int representation
  '''
  try: 
      int(s)
      return True
  except ValueError:
      return False

def turnPicsToSimpleInputs(artPath, jsonPath, cutoffSize=500, testProp=0.2):
  '''
  Turns card artwork into array representation and pairs each card with its onehot primary type
    encoding, separating training and test/validation sets
  Inputs:
    artPath: path to card art directory
    jsonPath: path to card info json file
    cutoffSize: minimum representaiton for at ype to be valid (500)
    testProp: proportion of art to separate from traingin for test/validation (0.2)
  Output:
    X: training art arrays
    Y: training category targets
    X_Test: testing art arrays
    Y_Test: testing category targets
    numCategories: total number of valid categories
  '''
  cardNameToCategories, numCategories, typeToCategory = generateCardToSimpleTypeDict(jsonPath, cutoffSize)

  X = []
  Y = []
  X_Test = []
  Y_Test = []

  artFiles = listdir(artPath)
  print('Generating Art Inputs')
  for index, art in enumerate(artFiles):
    if art == '.DS_Store':
      continue
    fileParts = art.split('.')
    if(representsInt(fileParts[0])):
      fileParts.pop(0)
    if fileParts[0][-1] == ' ':
      fileParts[0] = fileParts[0][:-1]
    if not fileParts[0] in cardNameToCategories:
      cardNameToCategories[fileParts[0]] = typeToCategory['Other']
    artPic = Image.open(artPath + art)
    artArray = np.array(artPic, dtype='float64')
    artData = artArray
    if random.random() < testProp:
      X_Test.append(artData)
      Y_Test.append(cardNameToCategories[fileParts[0]])
    else:
      X.append(artData)
      Y.append(cardNameToCategories[fileParts[0]])
    if int((index / len(artFiles)) * 10000) % 10 == 0:
      print ("Percent done: %2.1f %%" % ((index / len(artFiles)) * 100))
  print('Done')
  
  return (X,Y), (X_Test, Y_Test), numCategories

def getLiveDemoPicsToInput(artPath, cardPath, jsonPath, cutoffSize=500, numDesired=10,
                            showPics=False):
  '''
  Creates the data subset for a live demo of the convolutional network
  Inputs:
    artPath: path to card art
    cardPath: path to card scans
    jsonPath: path to card data json file
    cutoffSize: minimum representation for a primary type to be valid (500)
    numDesired: size of demo subset (10)
    showPics: boolean of whether or not to open card art/scans (False)
  Outputs:
    inputNames: array of names of cards in subset
    inputs: array representation of card art in subset
    numCategories: number of total valid type categories
    categoryToType: map from category number to correct type
    cardNameToCategories: map from card name to category
  '''
  cardNameToCategories, numCategories, typeToCategory = generateCardToSimpleTypeDict(jsonPath,
                                                                                      cutoffSize)
  inputNames = []
  inputs = []
  
  artFiles = listdir(artPath)
  subset = random.sample(artFiles, numDesired)
  print('Generating Live Demo Input Subset')
  for index, art in enumerate(subset):
    if art == '.DS_Store':
      continue
    fileParts = art.split('.')
    if(representsInt(fileParts[0])):
      fileParts.pop(0)
    if fileParts[0][-1] == ' ':
      fileParts[0] = fileParts[0][:-1]
    if not fileParts[0] in cardNameToCategories:
      cardNameToCategories[fileParts[0]] = typeToCategory['Other']
    artPic = Image.open(artPath + art)
    artArray = np.array(artPic, dtype='float64')
    artData = artArray
    inputNames.append(fileParts[0])
    inputs.append(artData)
    if showPics:
      cardPic = Image.open(cardPath + art)
      cardPic.show()
      artPic.show()
    if int((index / numDesired) * 10000) % 10 == 0:
      print ("Percent done: %2.1f %%" % ((index / numDesired) * 100))
  print('Done')

  categoryToType = dict((v,k) for k,v in typeToCategory.items())
  return inputNames, inputs, numCategories, categoryToType, cardNameToCategories

def simpleGenerateTypeSubtypeToNameInputs(jsonPath, maxLength=75):
  '''
  Generates input sequences for type, subtype, name generation
  Inputs:
    jsonPath: path to card data json file
    maxLength: maximum length for a sequence (75)
  Outputs:
    inputs: array of encoded sequences
    Outputs: array of encodings for the next character in sequence
    char_idx: map from chars in sequence to their ids in the encoding
    totalString: the complete string of card data, each line of which has the format
      'type1,type2;subtype1,subtype2;name'
  '''
  jsonFile = io.open(jsonPath)
  cardData = json.load(jsonFile)

  totalString = ''

  for cardName in cardData.keys():
    element = ''
    if 'types' in cardData[cardName]:
      for type in cardData[cardName]['types']:
        element+=type
        element+=','
      element= element[:-1] + ';'
    if 'subtypes' in cardData[cardName]:
      for subtype in cardData[cardName]['subtypes']:
        element+=subtype
        element+=','
      element = element[:-1] + ';'
    element+=cardName
    totalString+=element 
    totalString+='\n\n'

  return string_to_semi_redundant_sequences(totalString, maxLength), totalString
