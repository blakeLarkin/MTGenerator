def generateCardToTypeDict(jsonPath, cutoffSize=100):
  '''
  Creates a dictionary of card names to card types from a json file, only including types with a large enough representation. Creates a multihot relationship
  Inputs:
    jsonPath: path to magic the gather json file for card information
    cutoffSize: minimum size for a type to be included (100)
  Outputs:
    cardNameToCategories: dictionary from card names to the appropriate category of types
    numCategories: the total number of represented categories, not including the "other" category
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
  Turns card artwork into array representation and pairs each card with its multihot type encoding, separating training and test/validation sets
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
  paddedSequence = sequence[:desiredLength]
  paddedSequence.extend([0 for _ in range(desiredLength - len(paddedSequence))])
  return paddedSequence
