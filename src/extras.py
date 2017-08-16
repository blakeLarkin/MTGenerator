from os import listdir
from PIL import Image
import random
import io
import json
import numpy as np

def generateData(cardsPath, artPath='art', image_width=64, image_height=None, flip=True, blur=True,
                  grayscale=None, proportion=1):
  '''
  Generates card art for gan or convolutional network
  Inputs:
    cardsPath: path to card scans
    artPath: subpath to be appended to cardsPath after '../' to store card art ('art')
    image_width: final card art width in pixels (64)
    image_height: final card art height in pixels, if None will be set to image_width (None)
    flip: boolean for whether to store a flipped duplicate card art (True)
    blur: boolean for whether to store a blurred duplicate card art (True)
    grayscale: boolean for whether to save card art only as grayscale (False)
    proportion: proportion of card scans to get card art from
  '''
  if not image_height:
    image_height = image_width

  cards = getFiles(cardsPath)
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
    if grayscale:
      art = art.convert('L')
    art.save(cardsPath + '../' + artPath + '/' + cardPath)
    if flip:
      flip = art.transpose(Image.FLIP_LEFT_RIGHT)
      flip.save(cardsPath + '../' + artPath + '/flip_' + cardPath)
    if blur:
      blur = art.filter(ImageFilter.GaussianBlur(radius=0.6))
      blur.save(cardsPath + '../' + artPath + '/blur_' + cardPath)
      if flip:
        blurFlip = flip.filter(ImageFilter.GaussianBlur(radius=0.8))
        blurFlip.save(cardsPath + '../' + artPath + '/blur_flip_' + cardPath)
    if int((index / len(cards)) * 10000) % 10 == 0:
      print ("Percent done: %2.1f %%" % ((index / len(cards)) * 100))

def removeTokens(cardPath, jsonPath):
  '''
  Removes card scans of tokens
  Inputs:
    cardPath: path to card scans
    jsonPath: path to card data json file
  '''
  jsonFile = io.open(jsonPath)
  cardData = json.load(jsonFile)
  tokenNames = []
  for cardName in cardData.keys():
    if cardData[cardName]["layout"] == 'token':
      if cardName.split(" ")[-1] == 'card':
        tokenNames.append(cardName.split(" ")[0])
      else:
        tokenNames.append(cardName)

  cards = getFiles(cardPath)
  for card in cards:
    cardNameParts = card.split('.')
    if [part for part in cardNameParts if part in tokenNames]:
      remove(cardPath + card)

def removeSplits(cardPath, jsonPath):
  '''
  Removes card scans of splits
  Inputs:
    cardPath: path to card scans
    jsonPath: path to card data json file
  '''
  jsonFile = io.open(jsonPath)
  cardData = json.load(jsonFile)
  splitNames = []
  for cardName in cardData.keys():
    if cardData[cardName]["layout"] == 'split':
      splitNames.append(" - ".join(cardData[cardName]['names']))

  cards = getFiles(cardPath)
  for card in cards:
    cardNameParts = card.split('.')
    if [part for part in cardNameParts if part in splitNames]:
      remove(cardPath + card)

def generateCardToTypeDict(jsonPath, cutoffSize=100):
  '''
  Creates a dictionary of card names to card types from a json file, only including types with a
    large enough representation. Creates a multihot relationship
  Inputs:
    jsonPath: path to magic the gather json file for card information
    cutoffSize: minimum size for a type to be included (100)
  Outputs:
    cardNameToCategories: dictionary from card names to the appropriate category of types
    numCategories: the total number of represented categories
  '''
  jsonFile = io.open(jsonPath)
  cardData = json.load(jsonFile)

  numOfType = {}

  for cardName in cardData.keys():
    if 'types' in cardData[cardName]:
      for type in cardData[cardName]['types']:
        if not type in numOfType:
          numOfType[type] = 0
        numOfType[type]+=1

  typeToCategory = {}
  numCategories = 0

  for type in numOfType:
    if numOfType[type] > cutoffSize:
      typeToCategory[type] = numCategories
      numCategories+=1

  cardNameToCategories = {}

  for cardName in cardData.keys():
    category = [0 for _ in range(numCategories)]
    if 'types' in cardData[cardName]:
      for type in cardData[cardName]['types']:
        if type in typeToCategory:
          category[typeToCategory[type]] = 1
    cardNameToCategories[cardName] = category

  return (cardNameToCategories, numCategories)


def turnPicsToInputs(artPath, jsonPath, cutoffSize=500, testProp=0.2):
  '''
  Turns card artwork into array representation and pairs each card with its multihot type encoding,
    separating training and test/validation sets
  Inputs:
    artPath: path to card art directory
    jsonPath: path to card info json file
    cutoffSize: minimum representation for a type to be valid (500)
    testProp: proportion of art to separate from traingin for test/validation (0.2)
  Output:
    X: training art arrays
    Y: training category targets
    X_Test: testing art arrays
    Y_Test: testing category targets
  '''
  cardNameToCategories, numCategories = generateCardToTypeDict(jsonPath)

  X = []
  Y = []
  X_Test = []
  Y_Test = []

  artFiles = listdir(artPath)
  for art in artFiles:
    if art == '.DS_Store':
      continue
    fileParts = art.split('.')
    if(representsInt(fileParts[0])):
      fileParts.pop(0)
    if fileParts[0][-1] == ' ':
      fileParts[0] = fileParts[0][:-1]
    if not fileParts[0] in cardNameToCategories:
      cardNameToCategories[fileParts[0]] = [0 for _ in range(numCategories)]
    artPic = Image.open(artPath + art)
    artArray = np.array(artPic, dtype='float64')
    artData = artArray
    if random.random() < testProp:
      X_Test.append(artData)
      Y_Test.append(cardNameToCategories[fileParts[0]])
    else:
      X.append(artData)
      Y.append(cardNameToCategories[fileParts[0]])
  
  return (X,Y), (X_Test, Y_Test)

def generateTypeSubtypeToNameInputs(jsonPath, testProp=0.2):
  '''
  Generates padded sequences for a dynamic neural network
  Inputs:
    jsonPath: path to card data json file
    testProp: proportion of samples for test/validation
  Outputs:
    sequences: training input sequences
    testSequences: test/validation inpust sequences
    longestSequence: length of longest sequence
    dictionary: char to index mapping for encoding
  Note: this needs some work, it is unnecessarily complicated and doesn't fully encode sequences
  '''
  jsonFile = io.open(jsonPath)
  cardData = json.load(jsonFile)

  sequences = []
  testSequences = []
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
    if random.random() < testProp:
      testSequences.append(element)
    else:
      sequences.append(element)
    totalString+=element

  _, _, dictionary = string_to_semi_redundant_sequences(totalString)
  dictionary = dict((k, v+1) for k,v in dictionary.items())
  dictionary[''] = 0

  def mapToDict(sequence):
    encoding = []
    for char in sequence:
      encoding.append(dictionary[char])
    return encoding

  sequences = list(map(mapToDict, sequences))
  testSequences = list(map(mapToDict, testSequences))

  res = dict((v,k) for k,v in dictionary.items())

  longestSequence = 0
  for sequence in sequences:
    if len(sequence) > longestSequence and len(sequence) != 160:
      longestSequence = len(sequence)
  for sequence in testSequences:
    if len(sequence) > longestSequence and len(sequence) != 160:
      longestSequence = len(sequence)

  sequences = list(map(lambda seq: padding(seq, longestSequence), sequences))
  testSequences = list(map(lambda seq: padding(seq, longestSequence), testSequences))

  return sequences, testSequences, longestSequence, dictionary


def padding(sequence, desiredLength):
  '''
  Truncates or pads (with 0s at the end) a sequence to a desired length
  Inputs:
    sequence: sequence to be truncated or padded
    desiredLength: length of final sequence
  Outputs:
    paddedSequence: a sequence of desiredLength either truncated or padded
  '''
  paddedSequence = sequence[:desiredLength]
  paddedSequence.extend([0 for _ in range(desiredLength - len(paddedSequence))])
  return paddedSequence

def getCardTextSet(jsonPath, outputFile):
  '''
  Generate text file of sequences of the form type1,type2;subtype1;subtype2;cardText
  Inputs:
    jsonPath: path to card data json file
    outputFile: path to file to write final total sequence to
  '''
  jsonFile = io.open(jsonPath)
  cardData = json.load(jsonFile)

  cardEntries = ''

  for cardName in cardData.keys():
    cardEntry = ''
    if 'types' in cardData[cardName]:
      for type in cardData[cardName]['types']:
        cardEntry+=type
        cardEntry+=','
      cardEntry = cardEntry[:-1] + ';'
    if 'subtypes' in cardData[cardName]:
      for subtype in cardData[cardName]['subtypes']:
        cardEntry+=subtype
        cardEntry+=','
      cardEntry = cardEntry[:-1] + ';'
    if 'text' in cardData[cardName]:
      cardEntry+=cardData[cardName]['text']
    cardEntries+=(cardEntry)
    cardEntries+='\n\n'

  textFile = open(outputFile, 'w')
  textFile.write(cardEntries)
  textFile.close()
