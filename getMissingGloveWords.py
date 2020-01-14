import numpy as np
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform
from helper_utils import *
import re
import pickle as pk
missingWords = []
def sentences_to_indices(X, word_to_index, max_len):
	m = X.shape[0]  # number of training examples
	X_indices = np.zeros((m, max_len))
	for i in range(m):  # loop over training examples
		if i % 100 == 0:
			print("i = " + str(i) + " out of " + str(m))
		#sentence_words = (X[i].lower()).split()
		sentence_words = (X[i]).split()
		j = 0
		for w in sentence_words:
			try:
				X_indices[i, j] = word_to_index[w]
			except:
				if w not in missingWords:
					missingWords.append(w)
					print(w)
					if False and w == "BEST":
						print("i = " + str(i))
						print("X[i] = " + str(X[i]))
						print("j = " + str(j))
						exit()
			j = j + 1
	return X_indices

def pretrained_embedding_layer(word_to_vec_map, word_to_index):
	vocab_len = len(word_to_index) + 1  # adding 1 to fit Keras embedding (requirement)
	emb_dim = word_to_vec_map["THIS"].shape[0]  # define dimensionality of your GloVe word vectors (= 50)
	emb_matrix = np.zeros((vocab_len, emb_dim))
	for word, index in word_to_index.items():
		try:
			emb_matrix[index, :] = word_to_vec_map[word]
		except:
			print("failed here")
			print("word = " + str(word))
			print("index = " + str(index))
			print("word to vec:")
			print(word_to_vec_map[word])
			print("length of w3v:")
			print(str(len(word_to_vec_map[word])))
			emb_matrix[index, :] = word_to_vec_map[word][:50]
	embedding_layer = Embedding(vocab_len, emb_dim)
	embedding_layer.build((None,))
	embedding_layer.set_weights([emb_matrix])
	return embedding_layer
	
def SentimentAnalysis(input_shape, word_to_vec_map, word_to_index):
	sentence_indices = Input(shape=input_shape, dtype='int32')
	embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
	embeddings = embedding_layer(sentence_indices)
	X = LSTM(128, return_sequences=True)(embeddings)
	X = Dropout(0.5)(X)
	X = LSTM(128)(X)
	X = Dropout(0.5)(X)
	X = Dense(2, activation='softmax')(X)
	X = Activation('softmax')(X)
	model = Model(sentence_indices, X)
	return model

with open("amazonRefined.txt", "r") as f:
	trans = f.readlines()
trans = trans[1:]
trans = trans[:1000]

print("length of trans = " + str(len(trans)))
x = np.array([])
y = np.array([])

tIndex = 0
for t in trans:
	tIndex += 1
	if tIndex % 200 == 0:
		print("transcript " + str(tIndex) + " out of " + str(len(trans)))
	s = t.split("\t")
	xx = s[0]
	yy = int(s[1])
	x = np.append(x, xx)
	y = np.append(y, yy)

# Read 50 feature dimension glove file
print("reading glove file...")
word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('glove.6B.50d.txt')
#maxLen = len(max(x, key=len).split())
maxLen = 0
for xx in x:
	if len(xx.split()) > maxLen:
		maxLen = len(xx.split())
model = SentimentAnalysis((maxLen,), word_to_vec_map, word_to_index)
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
X_train_indices = sentences_to_indices(x, word_to_index, maxLen)

missingWords.sort()
print(missingWords)
for m in missingWords:
	print(m)
print("There are " + str(len(missingWords)) + " missing words.")
print("There are " + str(len(index_to_word)) + "  lines in the glove file")
print("Done.")
