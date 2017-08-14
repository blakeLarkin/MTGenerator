from os import listdir, remove
from PIL import Image, ImageFilter
import random
import json
import io

# TODO: reorganize files/functions, use list comprehension

def getFiles(path):
  return listdir(path)

def checkSizes(path):
  cards = getFiles(path)
  widthHeight = {}
  for cardPath in cards:
    if cardPath == '.DS_Store':
      continue
    card = Image.open(path + cardPath)
    width, height = card.size
    if not width in widthHeight:
      widthHeight[width] = height
      print(cardPath, width, height)

def generateData(cardsPath, artPath='art', image_width=64, image_height=None, flip=True, blur=True, grayscale=None, proportion=1):
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

def getTypesAndSubtypes(jsonPath):
  jsonFile = io.open(jsonPath)
  cardData = json.load(jsonFile)
  types = {}
  for cardName in cardData.keys():
    if 'types' in cardData[cardName]:
      for type in cardData[cardName]['types']:
        if not type in types:
          types[type] = []
        if 'subtypes' in cardData[cardName]:
          for subtype in cardData[cardName]['subtypes']:
            if not subtype in types[type]:
              types[type].append(subtype)

  print(types)

def getSubtypes(jsonPath):
  jsonFile = io.open(jsonPath)
  cardData = json.load(jsonFile)
  subtypes = ['none']
  for cardName in cardData.keys():
    if 'subtypes' in cardData[cardName]:
      for subtype in cardData[cardName]['subtypes']:
        if not subtype in subtypes:
          subtypes.append(subtype)

  print(subtypes)

def getCardTextSet(jsonPath, outputFile):
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

def generateCardToTypeSubtypeDict(jsonPath):
  jsonFile = io.open(jsonPath)
  cardData = json.load(jsonFile)

  numCategories = 0
  subtypeTypeToCategory = {}

  for cardName in cardData.keys():
    if 'subtypes' in cardData[cardName]:
      for subtype in cardData[cardName]['subtypes']:
        if not subtype in subtypeTypeToCategory:
          subtypeTypeToCategory[subtype] = numCategories
          numCategories+=1
    if 'types' in cardData[cardName]:
      for type in cardData[cardName]['types']:
        if not type in subtypeTypeToCategory:
          subtypeTypeToCategory[type] = numCategories
          numCategories+=1

  cardNameToCategories = {}

  for cardName in cardData.keys():
    category = [0 for _ in range(numCategories)]
    if 'subtypes' in cardData[cardName]:
      for subtype in cardData[cardName]['subtypes']:
        category[subtypeTypeToCategory[subtype]] = 1
    if 'types' in cardData[cardName]:
      for type in cardData[cardName]['types']:
        category[subtypeTypeToCategory[type]] = 1
    cardNameToCategories[cardName] = category

  print(cardNameToCategories)

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

  print(cardNameToCategories, typeToCategory)

