from os import listdir
from PIL import Image
import random
import io
import json
import numpy as np

from tflearn.data_utils import string_to_semi_redundant_sequences

def testPic(artPath):
  artFiles = listdir(artPath)
  for art in artFiles:
    artPic = Image.open(artPath + art)
    artData = np.array(artPic)
    print(artData)
    artData = artData.flatten()
    print(artData)
    break

def generatePics(cardsPath, artPath='art', image_width=64, image_height=None, proportion=1):
  if not image_height:
    image_height = image_width

  cards = listdir(cardsPath)
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

def generateCardToTypeDict(jsonPath):
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
    if numOfType[type] > 100:
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

def representsInt(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False

def turnPicsToInputs(artPath, jsonPath, testProp=0.2):
  cardNameToCategories, numCategories = generateCardToTypeDict(jsonPath)

  X = []
  Y = []
  X_Test = []
  Y_Test = []

  artFiles = listdir(artPath)
  for art in artFiles:
    fileParts = art.split('.')
    if(representsInt(fileParts[0])):
      fileParts.pop(0)
    if fileParts[0][-1] == ' ':
      fileParts[0] = fileParts[0][:-1]
    if not fileParts[0] in cardNameToCategories:
      cardNameToCategories[fileParts[0]] = [0 for _ in range(numCategories)]
      #print(fileParts[0])
    artPic = Image.open(artPath + art)
    artArray = np.array(artPic, dtype='float64')
    artData = artArray #.flatten()
    if random.random() < testProp:
      X_Test.append(artData)
      Y_Test.append(cardNameToCategories[fileParts[0]])
    else:
      X.append(artData)
      Y.append(cardNameToCategories[fileParts[0]])
  
  return (X,Y), (X_Test, Y_Test)

def generateCardToSimpleTypeDict(jsonPath, cutoffSize):
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

  cardNameToCategories = {}

  for cardName in cardData.keys():
    category = numCategories
    if 'types' in cardData[cardName]:
      if cardData[cardName]['types'][0] in typeToCategory:
        category = typeToCategory[cardData[cardName]['types'][0]]
    cardNameToCategories[cardName] = category

  return (cardNameToCategories, numCategories)

def turnPicsToSimpleInputs(artPath, jsonPath, cutoffSize=500, testProp=0.2):
  cardNameToCategories, numCategories = generateCardToSimpleTypeDict(jsonPath, cutoffSize)

  X = []
  Y = []
  X_Test = []
  Y_Test = []

  artFiles = listdir(artPath)
  for index, art in enumerate(artFiles):
    if art == '.DS_Store':
      continue
    fileParts = art.split('.')
    if(representsInt(fileParts[0])):
      fileParts.pop(0)
    if fileParts[0][-1] == ' ':
      fileParts[0] = fileParts[0][:-1]
    if not fileParts[0] in cardNameToCategories:
      cardNameToCategories[fileParts[0]] = numCategories
      #print(fileParts[0])
    artPic = Image.open(artPath + art)
    artArray = np.array(artPic, dtype='float64')
    artData = artArray #.flatten()
    if random.random() < testProp:
      X_Test.append(artData)
      Y_Test.append(cardNameToCategories[fileParts[0]])
    else:
      X.append(artData)
      Y.append(cardNameToCategories[fileParts[0]])
    if int((index / len(artFiles)) * 10000) % 10 == 0:
      print ("Percent done: %2.1f %%" % ((index / len(artFiles)) * 100))
  
  return (X,Y), (X_Test, Y_Test), numCategories

def padding(sequence, desiredLength):
  paddedSequence = sequence[:desiredLength]
  paddedSequence.extend([0 for _ in range(desiredLength - len(paddedSequence))])
  return paddedSequence

def generateTypeSubtypeToNameInputs(jsonPath, testProp=0.2):
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

def simpleGenerateTypeSubtypeToNameInputs(jsonPath, maxLength=75):
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

  return string_to_semi_redundant_sequences(totalString), totalString
