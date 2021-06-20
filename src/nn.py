import enum
from math import isnan
from nltk import downloader
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import tensorflow as tf
import src.cfg as cfg
from keras.models import model_from_json
from sklearn.metrics import precision_score, recall_score, f1_score


class NeuralNetwork:
	def __init__(self) -> None:
		self.df = pd.read_csv("dataset\spam_or_not_spam\spam_or_not_spam.csv")
		choice = int(input("Do you want to: \n1. Load the Word Model \n2. Train the Word Model\n"))
		self.__setup(choice)

	def __setup(self, choice) -> None:		
		nltk.download("punkt")
		nltk.download("stopwords")
		self.__set_emails()
		if choice == 2:
			self.__create_word_model()
		else:
			self.word_model = Word2Vec.load("models/word_model.bin")
		self.__apply_word2vec()
		self.__split_dataset()

	def __preprocess_dataframe(self) -> None:
		self.df.dropna(inplace=True)
		self.df.drop(self.df.tail(1).index,inplace=True)
		self.df = self.df.drop_duplicates()

	def __preprocess_text(self, text) -> list:
		tokens = word_tokenize(text)
		words = [word.lower() for word in tokens if word.isalnum()]
		stop_words = set(stopwords.words('english'))
		words = [word for word in words if not word in stop_words]
		return words

	def __set_emails(self) -> None:
		self.__preprocess_dataframe()
		emails = [email for email in self.df.email]
		self.tokenized_mails = []
		for mail in emails:
			words = self.__preprocess_text(mail)
			self.tokenized_mails.append(words)
	
	def __create_word_model(self) -> None:
		print("Starting word model training")
		self.word_model = Word2Vec(sentences=self.tokenized_mails, vector_size=200, min_count=1, workers=4)
		self.word_model.save('models/word_model.bin')
		print("Finished training!\n Model saved to models/word_model.bin")

	def __vectorize(self, wordlist) -> np.ndarray:
		vec = np.zeros((200,))
		for word in wordlist:
			wordvec = self.word_model.wv.get_vector(word)
			vec += wordvec
		vec /= len(wordlist)
		return vec

	def __apply_word2vec(self) -> None:
		self.df.email = self.df.email.apply(self.__preprocess_text)
		self.df.email = self.df.email.apply(self.__vectorize)
	

	def __create_nn_model(self) -> None:
		model = tf.keras.Sequential([
			tf.keras.layers.Dense(self.dataset_train.shape[0], activation='relu'),
			tf.keras.layers.Dropout(0.1),
			tf.keras.layers.Dense(self.dataset_train.shape[0], activation='relu'),
			tf.keras.layers.Dropout(0.1),
			tf.keras.layers.Dense(1, activation='sigmoid')
		])

		model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
		self.__nn_model = model


	def train_nn_model(self) -> None:
		self.__create_nn_model()
		x = np.array(self.dataset_train.email.tolist())
		y = np.array(self.dataset_train.label.tolist())
		x = tf.cast(x, tf.float32)
		y = tf.cast(y, tf.int32)
		size = self.dataset_train.shape[0]
		dataset = tf.data.Dataset.from_tensor_slices(
		(
			x, y
		))
		train_dataset = dataset.shuffle(size, reshuffle_each_iteration=True).batch(cfg.nn_batch_size)

		self.__nn_model.fit(train_dataset, epochs=cfg.nn_epochs, verbose=1, batch_size=cfg.nn_batch_size)

		print("Training Complete! Saving model to /models...")
		self.__savemodel()

	def test_nn(self) -> None:
		# evaluate loaded model on test data
		x = np.array(self.dataset_test.email.tolist())
		y = np.array(self.dataset_test.label.tolist())

		for i in x:
			for index, j in enumerate(i):
				if isnan(j):
					i[index] = 0.5
		xtf = tf.cast(x, tf.float32)
		ytf = tf.cast(y, tf.int32)

		y_predicted = self.__nn_model.predict(xtf)
		print(y_predicted)
		y_predicted = y_predicted.reshape(y_predicted.shape[0],1)
		y = y.reshape(y.shape[0],1)
		for num in y_predicted:
			print(num)
			if num[0] <= 0.5:
				num[0] = 0
			else: 
				num[0] = 1
		print(precision_score(y_predicted, y), recall_score(y_predicted, y), f1_score(y_predicted,y))

	def __savemodel(self) -> None:
		model_json = self.__nn_model.to_json()
		with open("models/nn_model.json", "w") as json_file:
			json_file.write(model_json)
			# serialize weights to HDF5
			self.__nn_model.save_weights("models/model.h5")
			print("Saved model to disk")
	
	def loadmodel(self) -> None:
		# load json and create model
		json_file = open('models/nn_model.json', 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		loaded_model = model_from_json(loaded_model_json)
		# load weights into new model
		loaded_model.load_weights("models/model.h5")
		print("Loaded model from disk")
		loaded_model.compile(optimizer='adam', loss='binary_crossentropy',  metrics=['accuracy'])
		self.__nn_model = loaded_model
	
	def __split_dataset(self) -> None:
		n_rows=self.df.shape[0]
		split_point = round(n_rows*0.75)
		self.dataset_train = self.df.iloc[:split_point,:]
		self.dataset_test = self.df.iloc[split_point:,:]
