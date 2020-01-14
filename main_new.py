useEmojiData = False
combineEmojiDataWithAmazonWay = False
useMentalData = True
combineEmojiTrainAndTest = False
#listOfDiseases = ['mdd', 'bpd', 'sz', 'psychosis']
listOfDiseases = ['mdd']

import numpy as np
if useEmojiData:
	if combineEmojiDataWithAmazonWay:
		from keras2.models import Model
		from keras2.layers import Dense, Input, Dropout, LSTM, Activation
		from keras2.layers.embeddings import Embedding
		from keras2.preprocessing import sequence
		from keras2.initializers import glorot_uniform
		from keras2 import optimizers
	else:
		from keras.models import Model
		from keras.layers import Dense, Input, Dropout, LSTM, Activation
		from keras.layers.embeddings import Embedding
		from keras.preprocessing import sequence
		from keras.initializers import glorot_uniform
		from keras import optimizers
else:
	from keras2.models import Model
	from keras2.layers import Dense, Input, Dropout, LSTM, Activation
	from keras2.layers.embeddings import Embedding
	from keras2.preprocessing import sequence
	from keras2.initializers import glorot_uniform
	from keras2 import optimizers
from helper_utils import *
import re
import pickle as pk
from sklearn.model_selection import train_test_split
import random

rawDataFileName = "amazonRefined.txt"
gloveFileName = 'glove.6B.50d.txt'

epochs=50
batch_size=50
useNumpyFile = False	#make True for faster
foldValidate = 3
testSizeCalc = 1.0 / foldValidate
randomSeed = 2
positiveValues = [5]
shouldLimitWordsPerTrans = False
wordsPerTrans = 9
maxWordsInTrans = 100	#maximum number of words in each transcript
#hiddenNodes1 = 512
hiddenNodes1 = 128
hiddenNodes2 = hiddenNodes1
learningRate=0.001
clipvalue=0.7
shouldLimitSizeOfData = True
dataSizeLimit = 300
shouldBalanceTrainingData = True

np.random.seed(randomSeed)

if useEmojiData:
	from emo_utils import *
	useNumpyFile = True
	gloveFileName = 'glove.6B.50d.txt_backup'
	shouldLimitWordsPerTrans = False
	shouldLimitSizeOfData = False
	#epochs=50
	epochs=20
	batch_size=32
	hiddenNodes1 = 128
	hiddenNodes2 = hiddenNodes1
	shouldBalanceTrainingData = False
	if combineEmojiTrainAndTest:
		rawDataFileName = 'emoji.txt'

if useMentalData:
	gloveFileName = 'mentalGlove.txt'
	useNumpyFile = False
	shouldLimitWordsPerTrans = False
	shouldLimitSizeOfData = False
	rawDataFileName = 'transcripts' + listOfDiseases[0] + '.csv'
	positiveValues = [1]
	hiddenNodes1 = 1024
	hiddenNodes2 = hiddenNodes1
	learningRate=0.0000001

def balanceTrainingData(train_x, train_y):
	#train_y must be all 1 or 0
	counts = [0, 0]
	for y in train_y:
		#counts[y] += 1
		if y in positiveValues:
			counts[1] += 1
		else:
			counts[0] += 1
	while counts[0] != counts[1]:
		if sum(counts) % 100 == 0:
			print("balancing training data: " + str(counts))
		majorityIsPositive = counts[1] > counts[0]
		ri = random.choice(range(len(train_x)))
		#if counts[train_y[ri]] <= counts[1 - train_y[ri]]:
		if (majorityIsPositive and train_y[ri] not in positiveValues) or (not majorityIsPositive and train_y[ri] in positiveValues):
			#train_x.append(train_x[ri])
			train_x = np.append(train_x, train_x[ri])
			#train_y.append(train_y[ri])
			train_y = np.append(train_y, train_y[ri])
			#counts[train_y[ri]] += 1
			if majorityIsPositive:
				counts[0] += 1
			else:
				counts[1] += 1
	return (train_x, train_y)

def sentences_to_indices(X, word_to_index, max_len):
	"""
	Converts an array of sentences (strings) into an array of indices corresponding to words in the sentences.
	The output shape should be such that it can be given to `Embedding()`
	
	Arguments:
	X -- array of sentences (strings), of shape (m, 1)
	word_to_index -- a dictionary containing the each word mapped to its index
	max_len -- maximum number of words in a sentence. You can assume every sentence in X is no longer than this.
	
	Returns:
	X_indices -- array of indices corresponding to words in the sentences from X, of shape (m, max_len)
	"""
	
	m = X.shape[0]  # number of training examples
	"""
	biggestFirstShape = -1
	biggestSecondShape = -1
	biggestShapeIndex = -1
	print("length of X is " + str(len(X)))
	print("length of X[0] is " + str(len(X[0])))
	print("length of X[0][0] is " + str(len(X[0][0])))
	print("X[0][0] is " + str(X[0][0]))
	print("X[0] is " + str(X[0]))
	print("length of X.shape = " + str(len(X.shape)))
	print("X.shape = " + str(X.shape))
	print("X.shape[0] = " + str(X.shape[0]))
	"""
	print("m = " + str(m))
	print("max_len = " + str(max_len))
	print("calculating new max_len...")
	for i in range(m):
		sentence_words = X[i].split()
		if len(sentence_words) > max_len:
			max_len = len(sentence_words)
			print("$$$ new max_len = " + str(max_len) + " $$$")
	#exit()
	# Initialize X_indices as a numpy matrix of zeros and the correct shape (≈ 1 line)
	X_indices = np.zeros((m, max_len))
	
	for i in range(m):  # loop over training examples
		# Convert the ith training sentence in lower case and split is into words. You should get a list of words.
		sentence_words = (X[i].upper()).split()
		if useEmojiData:
			sentence_words = (X[i].lower()).split()
		# Initialize j to 0
		j = 0
		# Loop over the words of sentence_words
		for w in sentence_words:
			# Set the (i,j)th entry of X_indices to the index of the correct word.
			try:
				X_indices[i, j] = word_to_index[w]
			except:
				print("Warning...unknown word.  Prematurly terminating application.")
				print("i = " + str(i))
				print("j = " + str(j))
				print("shape of X_indices = " + str(np.shape(X_indices)))
				print(w)
				#print("word_to_index first 10")
				#print(word_to_index[:10])
				#for www in word_to_index[:10]:
				#	print(www)
				print("length of word_to_index = " + str(len(word_to_index)))
				print("number of words in sentence = " + str(len(sentence_words)))
				print("sentence:")
				print(sentence_words)
				print("X[0]:")
				print(X[0])
				#X_indices[i, j] = -1
				exit()
				pass
			# Increment j to j + 1
			j = j + 1
	return X_indices

def pretrained_embedding_layer(word_to_vec_map, word_to_index):
	"""
	Creates a Keras Embedding() layer and loads in pre-trained GloVe 50-dimensional vectors.

	Arguments:
	word_to_vec_map -- dictionary mapping words to their GloVe vector representation.
	word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

	Returns:
	embedding_layer -- pretrained layer Keras instance
	"""
	vocab_len = len(word_to_index) + 1  # adding 1 to fit Keras embedding (requirement)
	emb_dim = -1
	if useEmojiData:
		emb_dim = word_to_vec_map["cucumber"].shape[0]
	else:
		if useMentalData:
			emb_dim = word_to_vec_map["THE"].shape[0]
		else:
			emb_dim = word_to_vec_map["THIS"].shape[0]  # define dimensionality of your GloVe word vectors (= 50)

	# Initialize the embedding matrix as a numpy array of zeros of shape (vocab_len, dimensions of word vectors = emb_dim)
	emb_matrix = np.zeros((vocab_len, emb_dim))

	# Set each row "index" of the embedding matrix to be the word vector representation of the "index"th word of the vocabulary
	for word, index in word_to_index.items():
		emb_matrix[index, :] = word_to_vec_map[word]

	# Define Keras embedding layer with the correct output/input sizes, make it trainable. Use Embedding(...). Make sure to set trainable=False.
	embedding_layer = Embedding(vocab_len, emb_dim)

	# Build the embedding layer, it is required before setting the weights of the embedding layer. Do not modify the "None".
	embedding_layer.build((None,))

	# Set the weights of the embedding layer to the embedding matrix. Your layer is now pretrained.
	embedding_layer.set_weights([emb_matrix])

	return embedding_layer


def SentimentAnalysis(input_shape, word_to_vec_map, word_to_index):
	"""
	Function creating the Emojify-v2 model's graph.

	Arguments:
	input_shape -- shape of the input, usually (max_len,)
	word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
	word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

	Returns:
	model -- a model instance in Keras
	"""
	# Define sentence_indices as the input of the graph, it should be of shape input_shape and dtype 'int32' (as it contains indices).
	#sentence_indices = Input(shape=input_shape, dtype=np.int32)
	sentence_indices = Input(shape=input_shape, dtype='int32')

	# Create the embedding layer pretrained with GloVe Vectors (≈1 line)
	embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)

	# Propagate sentence_indices through your embedding layer, you get back the embeddings
	embeddings = embedding_layer(sentence_indices)

	# Propagate the embeddings through an LSTM layer with 128-dimensional hidden state
	# Be careful, the returned output should be a batch of sequences.
	X = LSTM(hiddenNodes1, return_sequences=True)(embeddings)
	# Add dropout with a probability of 0.5
	X = Dropout(0.5)(X)
	# Propagate X trough another LSTM layer with 128-dimensional hidden state
	# Be careful, the returned output should be a single hidden state, not a batch of sequences.
	X = LSTM(hiddenNodes2)(X)
	# Add dropout with a probability of 0.5
	X = Dropout(0.5)(X)
	# Propagate X through a Dense layer with softmax activation to get back a batch of 5-dimensional vectors.
	if useEmojiData:
		X = Dense(5, activation='softmax')(X)
	else:
		X = Dense(2, activation='softmax')(X)
	# Add a softmax activation
	X = Activation('softmax')(X)

	# Create Model instance which converts sentence_indices into X.
	model = Model(sentence_indices, X)

	return model


if __name__ == "__main__":
	
	#delete this:::
	#word_to_index, index_to_word, word_to_vec_map = read_glove_vecs(gloveFileName)
	#print(index_to_word)
	#exit(0)
	"""
	print(str(len(index_to_word)))
	outList = []
	for index in range(1, len(index_to_word) + 1):
		if index % 1000 == 0:
			print("Processing " + str(index) + " out of " + str(len(index_to_word)))
		outList.append(index_to_word[index])
	print("Pickling word list...")
	with open("dictionary.pkl", "wb") as f:
		pk.dump(outList, f)
	exit()
	"""
	
	
	# Read train and test files
	#X_train, Y_train = read_csv('train_emoji.csv')
	#X_test, Y_test = read_csv('test_emoji.csv')
	#maxLen = len(max(X_train, key=len).split())
	
	#print("len x_train = " + str(len(X_train)))
	#print("len y_train = " + str(len(Y_train)))
	#print(Y_train)
	#print(X_train[:10])
	
	x = []
	y = []
	
	if useNumpyFile:
		if useEmojiData:
			if combineEmojiTrainAndTest:
				print("reading combined emoji files...")
				with open(rawDataFileName, "r") as f:
					trans = f.readlines()
				for i in range(len(trans)):
					trans[i] = trans[i].upper()
				random.shuffle(trans)
				print("length of trans = " + str(len(trans)))
				x = np.array([])
				y = np.array([])
				print("Processing transcripts...")
				tIndex = 0
				for t in trans:
					tIndex += 1
					if tIndex % 10 == 0:
						print("transcript " + str(tIndex) + " out of " + str(len(trans)))
						if shouldLimitSizeOfData:
							print(str(len(x)) + " are in data with a limit of " + str(dataSizeLimit))
					s = t.split("\t")
					xx = re.sub(r'([^\s\w]|_)+', '', s[0])
					yy = int(s[1])
					x = np.append(x, xx)
					y = np.append(y, yy)
				y = y.astype(int)
				X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=testSizeCalc, random_state=randomSeed, stratify=y)
			else:
				print("reading emoji files")
				# Read train and test files
				X_train, Y_train = read_csv('train_emoji.csv')
				X_test, Y_test = read_csv('test_emoji.csv')
			# Convert one-hot-encoding type, classification =5, [1,0,0,0,0]
			try:
				Y_oh_train = convert_to_one_hot(Y_train, C=5)
			except:
				print("Y_train failed, here is y_train:")
				print(Y_train)
				exit(1)
			Y_oh_test = convert_to_one_hot(Y_test, C=5)
		else:
			print("reading transX.pkl...")
			with open("transX.pkl", "rb") as f:
				x = pk.load(f)
			print("reading transY.pkl...")
			with open("transY.pkl", "rb") as f:
				y = pk.load(f)
			print("reading complete")
	else:
		print("Not using numpy file")
		with open(rawDataFileName, "r") as f:
			trans = f.readlines()
		trans = trans[1:]
		if useMentalData:
			for i in range(len(trans)):
				trans[i] = trans[i].upper()
		random.shuffle(trans)
		
		print("length of trans = " + str(len(trans)))
		x = np.array([])
		y = np.array([])
		print("Processing transcripts...")
		tIndex = 0
		#print("transcript:")
		#print(trans)
		for t in trans:
			tIndex += 1
			if tIndex % 10 == 0:
				print("transcript " + str(tIndex) + " out of " + str(len(trans)))
				if shouldLimitSizeOfData:
					print(str(len(x)) + " are in data with a limit of " + str(dataSizeLimit))
			s = t.split("\t")
			xx = re.sub(r'([^\s\w]|_)+', '', s[0])
			yy = int(s[1])
			"""
			x.append(xx)
			y.append(yy)
			"""
			#if shouldLimitWordsPerTrans and len(xx.split()) <= maxWordsInTrans:
			if not shouldLimitWordsPerTrans or len(xx.split()) == wordsPerTrans:
				x = np.append(x, xx)
				y = np.append(y, yy)
				#tIndex -= 1
			if shouldLimitSizeOfData and len(y) == dataSizeLimit:
				break
		print("dumping transX...")
		with open("transX.pkl", "wb") as f:
			pk.dump(x, f)
		print("dumping transY...")
		with open("transY.pkl", "wb") as f:
			pk.dump(y, f)
		print("dumping finished")
	
	print("len x_train = " + str(len(x)))
	print("len y_train = " + str(len(y)))
	#print(y)
	
	"""
	print("converting x to npArray")
	np.asarray(x)
	print("converting Y to npArray")
	np.asarray(y)
	"""
	
	#print(x[:10])
	"""
	print("type of trainX = " + str(type(X_train)))
	print("type of x = " + str(type(x)))
	print("type of trainY = " + str(type(Y_train)))
	print("type of y = " + str(type(y)))
	"""
	
	if not useEmojiData:
		X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=testSizeCalc, random_state=randomSeed, stratify=y)
	
	#maxLen = len(max(X_train, key=len).split())  #<-----bug in original code.  Corrected in following lines
	maxLen = 0
	for xx in X_train:
		if len(xx.split()) > maxLen:
			maxLen = len(xx.split())
	
	if shouldBalanceTrainingData:
		(X_train, Y_train) = balanceTrainingData(X_train, Y_train)
		
	"""		
	lenLst = []
	for xx in x:
		lenLst.append(len(xx.split()))
	print("average = " + str(sum(lenLst) / len(lenLst)))
	print(lenLst)
	exit()
	"""
	if useEmojiData:
		# Convert one-hot-encoding type, classification =5, [1,0,0,0,0]
		Y_oh_train = convert_to_one_hot(Y_train, C=5)
		Y_oh_test = convert_to_one_hot(Y_test, C=5)
	else:
		# Convert one-hot-encoding type, classification =5, [1,0,0,0,0]
		#Y_oh_train = convert_to_one_hot(Y_train, C=5)
		#Y_oh_test = convert_to_one_hot(Y_test, C=5)
		#"""
		#Y_oh_train = np.array([])
		#Y_oh_test = np.array([])
		Y_oh_train = []
		Y_oh_test = []
		print("Processing Y_train...")
		yIndex = 0
		testYes = 0
		testNo = 0
		trainYes = 0
		trainNo = 0
		for yy in Y_train:
			yIndex += 1
			if yIndex % 10000 == 0:
				print(str(yIndex) + " out of " + str(len(Y_train)))
			if yy in positiveValues:
				#Y_oh_train = np.append(Y_oh_train, [np.array([0., 1.])])
				Y_oh_train.append([0., 1.])
				trainYes += 1
			else:
				#Y_oh_train = np.append(Y_oh_train, [np.array([1., 0.])])
				Y_oh_train.append([1., 0.])
				trainNo += 1
		print("Processing Y_test...")
		yIndex = 0
		for yy in Y_test:
			yIndex += 1
			if yIndex % 10000 == 0:
				print(str(yIndex) + " out of " + str(len(Y_test)))
			if yy in positiveValues:
				#Y_oh_test = np.append(Y_oh_test, [np.array([0., 1.])])
				Y_oh_test.append([0., 1.])
				testYes += 1
			else:
				#Y_oh_test = np.append(Y_oh_test, [np.array([1., 0.])])
				Y_oh_test.append([1., 0.])
				testNo += 1
		#"""
		print("trnYes\ttrnNo\ttestYes\ttestNo")
		for aa in [trainYes, trainNo, testYes, testNo]:
			print(str(aa) + "\t", end = '')
		print("\n")	
		print("percent yes = " + str(100 * (trainYes + testYes) / (trainYes + trainNo + testYes + testNo)))
	
	if not useEmojiData:
		print("converting Y_oh_train to np...")
		Y_oh_train = np.array(Y_oh_train)
		print("convert Y_oh_test to np...")
		Y_oh_test = np.array(Y_oh_test)
	
	# Read 50 feature dimension glove file
	print("reading glove file...")
	word_to_index, index_to_word, word_to_vec_map = read_glove_vecs(gloveFileName)
	
	# Model and model summmary
	print("maxLen = " + str(maxLen))
	model = SentimentAnalysis((maxLen,), word_to_vec_map, word_to_index)
	model.summary()
	#exit(0)
	#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	sgd = optimizers.SGD(lr=learningRate, clipvalue=clipvalue)
	if useEmojiData:
		sgd = 'adam'
	model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
	
	with open("modelBeforeTraining.pkl", "wb") as f:
		print("dumping model before training...")
		pk.dump(model, f)
		
	print("There are " + str(len(model.layers)) + " layers:")
	layerNumber = 0
	for ll in model.layers:
		layerNumber += 1
		print("Layer number " + str(layerNumber))
		weights = ll.get_weights()
		print("There are " + str(len(weights)) + " weights")
		print("The shape of the weights is:")
		print(np.shape(weights))
		wNum = 0
		for ww in weights:
			wNum += 1
			print("Weight number " + str(wNum) + ":")
			print(np.shape(ww))
			print(ww)
			"""
			for www in ww:
				print(www)
			"""
		#print(weights)
	#print("Exiting prematurely.")
	#exit()
	"""
	print("\n\n\nDigging deeper before:")
	ll = model.layers[6]
	weights = ll.get_weights()
	print("There are " + str(len(weights)) + " weights")
	print("The shape of the weights is:")
	print(np.shape(weights))
	
	#print("Exiting prematurely")
	#exit(0)
	for mm in weights:
		print(len(mm))
	print("the weights are:")
	for mm in weights:
		print(mm)
	"""
	print("creating X_train_indices...")
	print("X_train[:10] = " + str(X_train[:10]))
	X_train_indices = sentences_to_indices(X_train, word_to_index, maxLen)
	#Y_train_oh = convert_to_one_hot(Y_train, C=5)
	
	if False:
		print("\n\n\nDigging deeper before:")
		for layerIndex in range(len(model.layers)):
			ll = model.layers[layerIndex]
			weights = ll.get_weights()
			print("There are " + str(len(weights)) + " weights")
			print("The shape of the weights is:")
			print(np.shape(weights))
			
			#print("Exiting prematurely")
			#exit(0)
			for mm in weights:
				print(len(mm))
			print("the weights are:")
		
			for indWeightIndex in range(len(weights)):
				#indWeightIndex = 0
				print("\n\nIndividual weights:")
				indWeight = weights[indWeightIndex]
				print("Individual weight number " + str(indWeightIndex) + ":")
				print("Shape: " + str(np.shape(indWeight)))
				#print("Weight proper:")
				#print(indWeight)	
			
	#print("Exiting prematurely")
	#exit(0)

	# Train model
	#print(Y_oh_train)
	X_test_indices = sentences_to_indices(X_test, word_to_index, maxLen)
	#print("Running model.evalueate on test data before training...")
	#loss, acc = model.evaluate(X_test_indices, Y_oh_test)
	#print()
	#print("Test accuracy = ", acc)
	#print("Running model.predict on test data...")
	#pred = model.predict(X_test_indices)
	 
	#print("Stopping before we have a fit.")
	#exit()
	print("Having a fit...")
	try:
		model.fit(X_train_indices, Y_oh_train, epochs=epochs, batch_size=batch_size, shuffle=True)
	except KeyboardInterrupt:
		print("Training stopped prematurely by user")
		pass
	print("Training stopped")
	
	print("exiting prematurely after having a fit.")
	#exit()
	
	with open("modelAfterTraining.pkl", "wb") as f:
		print("dumping model after training...")
		pk.dump(model, f)
		
	
	print("Digging deeper after:")
	ll = model.layers[6]
	weights = ll.get_weights()
	print("There are " + str(len(weights)) + " weights")
	print("The shape of the weights is:")
	print(np.shape(weights))
	for mm in weights:
		print(len(mm))
	print("the weights are:")
	for mm in weights:
		print(mm)
	"""
	if True:
		for indWeightIndex in range(len(weights)):
			#indWeightIndex = 0
			print("\n\nIndividual weights:")
			indWeight = weights[indWeightIndex]
			print("Individual weight number " + str(indWeightIndex) + ":")
			print("Shape: " + str(np.shape(indWeight)))
			print("Weight proper:")
			print(indWeight)	
	print("Exiting prematurely")
	exit()
	"""
	X_test_indices = sentences_to_indices(X_test, word_to_index, max_len=maxLen)
	#Y_test_oh = convert_to_one_hot(Y_test, C=2)

	# Evaluate model, loss and accuracy
	#loss, acc = model.evaluate(X_test_indices, Y_test_oh)
	
	print("X_test_indices = :")
	print(X_test_indices)
	print("Running model.evalueate on test data...")
	loss, acc = model.evaluate(X_test_indices, Y_oh_test)
	print()
	print("Test accuracy = ", acc)

	# Compare prediction and expected emoji
	C = 2
	if useEmojiData:
		C = 5
	#y_test_oh = np.eye(C)[Y_test.reshape(-1)]
	X_test_indices = sentences_to_indices(X_test, word_to_index, maxLen)
	print("Running model.predict on test data...")
	pred = model.predict(X_test_indices)
	#exit()
	print("List of incorrect test predictions:")
	for i in range(min(len(X_test), 20)):
		x = X_test_indices
		num = np.argmax(pred[i])
		if (num != Y_test[i]):
			print(' prediction: '  + str(num) + "\t" + X_test[i])
	"""
	print("Looking for 0 predictions in test data:")
	for i in range(len(X_test)):
		num = np.argmax(pred[i])
		if num == 0:
			print(X_test[i])
	print("That was all of the 0 predictions for test data.\n\n")
	
	print("Looking for 0 predictions in training data:")
	pred = model.predict(X_train_indices)
	for i in range(len(X_test)):
		num = np.argmax(pred[i])
		if num == 0:
			print(X_test[i])
	print("That was all of the 0 predictions for training data.")
	"""

	# Test your sentence
	"""
	x_test = np.array(['she is a beauty'])
	X_test_indices = sentences_to_indices(x_test, word_to_index, maxLen)
	print(x_test[0] + ' ' + label_to_emoji(np.argmax(model.predict(X_test_indices))))
	"""
