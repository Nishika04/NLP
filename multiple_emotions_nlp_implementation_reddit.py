# -*- coding: utf-8 -*-
"""Multiple_emotions_NLP_Implementation_Reddit.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1lH55go3nm7zefj8EQ2G4bfSErMCwNDJe

### **Dataset: GoEmotions**

*GoEmotions* is a corpus of 58k carefully curated comments extracted from Reddit, with human annotations to 27 emotion categories or Neutral.

Number of examples: 58,009.
Number of labels: 27 + Neutral.

The data has already been separated in train, test and validation sets

Size of training dataset: 43,410.

Size of test dataset: 5,427.

Size of validation dataset: 5,426.

The emotion categories are: admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, optimism, pride, realization, relief, remorse, sadness, surprise.

Importing libraries and importing data
"""


import tensorflow as tf
print(tf.__version__)

# Here we import everything we need for the project

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
#import cv2
import pandas as pd


#Importing translator library
from googletrans import Translator

# Sklearn
from sklearn.model_selection import train_test_split # Helps with organizing data for training
from sklearn.metrics import confusion_matrix # Helps present results as a confusion-matrix

print(tf.__version__)

# Importing raw comments and labels
df_train = pd.read_csv('train.tsv', sep='\t',nrows=None, header=None, names=['Text', 'GE_indices', 'Id']).drop('Id', axis=1)

print(df_train.shape)

df_val = pd.read_csv('dev.tsv', sep='\t', header=None, names=['Text', 'GE_indices', 'Id']).drop('Id', axis=1)
df_test = pd.read_csv('test.tsv', sep='\t', header=None, names=['Text', 'GE_indices', 'Id']).drop('Id', axis=1)

# Preview of what data has been saved in the above variables
df_train[:10]

df_test

df_val

"""Finding out how much percentage of data is represented by each"""

# Defining the number of samples in train, validation and test dataset
size_train = df_train.shape[0]
size_val = df_val.shape[0]
size_test = df_test.shape[0]

# Defining the total number of samples
size_all = size_train + size_val + size_test

# Shape of train, validation and test datasets
print("Train dataset has {} samples and represents {:.2f}% of overall data".format(size_train, size_train/size_all*100))
print("Validation dataset has {} samples and represents {:.2f}% of overall data".format(size_val, size_val/size_all*100))
print("Test dataset has {} samples and represents {:.2f}% of overall data".format(size_test, size_test/size_all*100))
print()
print("The total number of samples is : {}".format(size_all))

"""Finding out number of emotions in the dataset"""

#For datatype
df_train.info()

"""Now, emotions.txt has all the set of emotions that this dataset contains.

Each emotion is mapped to a number as well, according to the sequence it is in, in the emotions.txt text file
"""

# Loading emotion labels for classification of emotions in GoEmotions dataset
with open("emotions.txt", "r") as file:
    GE_class = file.read().split("\n")

print(GE_class)

GE_class[25]

print(len(GE_class))

"""Hence, there are 28 emotions, including 'neutral'"""

#Counting the number of values under each emotion category
df_train.GE_indices.value_counts()

"""###Lambda Functions

![image.png](data:image/png;base64,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)

They can have any number of arguments but only one expression.

Before we start to further process the data, we will first have to combine all the three datasets into one and then proceed further
"""

# Combining all the 3 datasets into 1 dataset
df_all = pd.concat([df_train, df_val, df_test], axis=0).reset_index(drop=True)

# Preview of data
print(df_all.head(3))

print(df_all.shape)

"""Next, we will have to map each GE_indices to it's corresponding emotion in the emotions.txt file

For that, let's first convert these GE_indices into a list of indices which we will then map to the emotions.txt file and extract the emotion labels from the file
"""

# We will use a lambda function here to make things easier
#df_all['GE_indices'] = df_all['GE_indices'].apply(lambda x: x.split(','))

# Viewing the data for results
print(df_all.head)

"""Next steps:
1. Determine which emotions we want
2. Drop accordingly
3. Tokenization
4. Embedding
5. Vectorization
6. Models building
"""

df_all.isnull().sum()

df_all

"""Dropping the un required emotions"""


# 2:anger, 9:disappointment, 14:fear, 17:joy, 26:surprise, 27:neutral
df_final2 = pd.DataFrame()
emotions_selected=['2','9','14','17','26','27']#

for i in range(6):
   df_final2=df_final2._append(df_all.loc[df_all['GE_indices'] == emotions_selected[i]])
   print(emotions_selected[i])

print (df_final2)



df_final2.head()

one_hot_encoded_data = pd.get_dummies(df_final2, columns = ['GE_indices'])

one_hot_encoded_data.dtypes

one_hot_encoded_data.head()

df_final2.GE_indices.value_counts()

X = df_final2.iloc[:, 0].values
Y = one_hot_encoded_data.iloc[:, 1:7].values

X

Y

print(X.shape)
print(Y.shape)

"""###Split data  into training and validation sets

"""

from sklearn.model_selection import train_test_split

train_sentences, val_sentences, train_labels, val_labels= train_test_split(X,
                                                                           Y,
                                                                           test_size=0.1,#use 20%of training data for validation
                                                                           random_state=42)

len(train_sentences),len(train_labels),len(val_sentences), len(val_labels)

#checking the samples
train_sentences[:10], train_labels[:10]

"""##converting text to numbers
When dealing with text we needd to convert it into numbers

there are a few ways to do this
* Tokenisation- direct mapping of token(a token could be word or a character) to number
* Embedding - create a matrix of feature vector for each token (size of the feature vector can be defined and this embedding can be learned)
"""

import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization



#using default textvectorization parameters
text_vectorizer = TextVectorization(max_tokens=None, # how many words in the vocabulary (all of the different words in your text)
                                    standardize="lower_and_strip_punctuation", # how to process text
                                    split="whitespace", # how to split tokens
                                    ngrams=None, # create groups of n-words?
                                    output_mode="int", # how to map tokens to numbers
                                    output_sequence_length=None) # how long should the output sequence of tokens be?
                                    # pad_to_max_tokens=True)9

#find the average num of tokens in the training tweets
round(sum([len(i.split())for i in train_sentences ])/len(train_sentences))

#setup text vectorization variables
max_vocab_length=10000 # max num of words to have in our vocabulary
max_length= 15 #max length our sequences will be (eg how many words from a tweet does a model need to see)\

text_vectorizer=TextVectorization(max_tokens=max_vocab_length,
                                  output_mode="int",
                                  output_sequence_length=max_length)

#fit the text vecctorizer to the training text
text_vectorizer.adapt(train_sentences)

sample_sentence="Wow we are finally implenting! aren't we great"
text_vectorizer([sample_sentence])

import random
random_sentence=random.choice(train_sentences)
print(f"Original text; \n {random_sentence}\
        \n\n Vectorized version:")
text_vectorizer([random_sentence])

# Get the unique words in vocabulary
words_in_vocab= text_vectorizer.get_vocabulary()
top_5_words= words_in_vocab[:5]#get 5 most common
bottom_5_words= words_in_vocab[-5:]#get 5 least common
print(f"Number of words in vocab:{len(words_in_vocab)}")
print(f"5 most common words in vocab:{top_5_words}")
print(f"5 most least words in vocab:{bottom_5_words}")

"""##Creating embedding layer
using tensors flows embeding layer
"""

tf.random.set_seed(42)
from tensorflow.keras import layers

embedding = layers.Embedding(input_dim=max_vocab_length, # set input shape
                             output_dim=128, # set size of embedding vector
                             embeddings_initializer="uniform", # default, intialize randomly
                             input_length=max_length, # how long is each input
                             name="embedding_1")

embedding

# Get a random sentence from training set
random_sentence = random.choice(train_sentences)
print(f"Original text:\n{random_sentence}\
      \n\nEmbedded version:")

# Embed the random sentence (turn it into numerical representation)
sample_embed = embedding(text_vectorizer([random_sentence]))
sample_embed

# Check out a single token's embedding
sample_embed[0][0], sample_embed[0][0].shape, random_sentence[0]

"""###Modelling a text dataset (running a series of experiements )

### Creating an evaluation function for our model experiments

We can evaluate all of our model's predictions with different metrics every single time, but that will become very cumbersome.

For this, we will create a function.

Let's create one to compare our model's predictions with the truth labels using the following metrics: accuracy, precision, recall, f1 score
"""



import wget
url="https://raw.githubusercontent.com/mrdbourke/tensorflow-deep-learning/main/extras/helper_functions.py"
filename=wget.download(url)

# Download helper functions script
#wget "https://raw.githubusercontent.com/mrdbourke/tensorflow-deep-learning/main/extras/helper_functions.py"
#import series of helper functions
from helper_functions import unzip_data, create_tensorboard_callback, plot_loss_curves, compare_historys

# Function to evaluate: accuracy, precision, recall, f1 score

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def calculate_results(y_true, y_pred):
  """
  Calculates model accuracy, precision, recall and f1 score of a binary classification model.

  Parameters:
  -----
  y_true = true labels in the form of a 1D array
  y_pred = predicted labels in the form of a 1D array

  Returns a dictionary of accuracy, precision, recall, f1-score.
  """
  # Calculate model accuracy
  model_accuracy = accuracy_score(y_true, y_pred) * 100
  # Calculate model precision, recall and f1 score using "weighted" average
  model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
  model_results = {"accuracy": model_accuracy,
                  "precision": model_precision,
                  "recall": model_recall,
                  "f1": model_f1}
  return model_results
  # _ is used as an blank variable because we dont want 'support' which is a type of return in the function

"""##Model 1: A simple dense model
The first "deep" model we're going to build is a single layer dense model. In fact, it's barely going to have a single layer.

It'll take our text and labels as input, tokenize the text, create an embedding, find the average of the embedding (using Global Average Pooling) and then pass the average through a fully connected layer with one output unit and a sigmoid activation function.

If the previous sentence sounds like a mouthful, it'll make sense when we code it out (remember, if in doubt, code it out).

And since we're going to be building a number of TensorFlow deep learning models, we'll import our create_tensorboard_callback() function from helper_functions.py to keep track of the results of each.

"""

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# Create tensorboard callback (need to create a new one for each model)
from helper_functions import create_tensorboard_callback

# Create directory to save TensorBoard logs
SAVE_DIR = "model_logs"

# Build model with the Functional API
from tensorflow.keras import layers
inputs = layers.Input(shape=(1,), dtype="string") # inputs are 1-dimensional strings
x = text_vectorizer(inputs) # turn the input text into numbers
x = embedding(x) # create an embedding of the numerized numbers
x = layers.GlobalAveragePooling1D()(x) # lower the dimensionality of the embedding (try running the model without this layer and see what happens)
outputs = layers.Dense(6, activation="softmax")(x) # create the output layer, want binary outputs so use sigmoid activation
model_1 = tf.keras.Model(inputs, outputs, name="model_1_dense") # construct the model

# Compile model
model_1.compile(loss="CategoricalCrossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

# Get a summary of the model
model_1.summary()

train_sentences_list = train_sentences.tolist()

val_sentences_list = val_sentences.tolist()

print(train_sentences.shape)

type(train_labels)

print(train_labels.shape)

len(train_labels)

train_labels[0]='1'

train_labels

# train_labels2

# val_labels2

"""###Model 4: Bidirectonal RNN model"""

# Set random seed and create embedding layer (new embedding layer for each model)
tf.random.set_seed(42)
from tensorflow.keras import layers
model_4_embedding = layers.Embedding(input_dim=max_vocab_length,
                                     output_dim=128,
                                     embeddings_initializer="uniform",
                                     input_length=max_length,
                                     name="embedding_4")

# Build a Bidirectional RNN in TensorFlow
inputs = layers.Input(shape=(1,), dtype="string")
x = text_vectorizer(inputs)
x = model_4_embedding(x)
# x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x) # stacking RNN layers requires return_sequences=True
x = layers.Bidirectional(layers.LSTM(64))(x) # bidirectional goes both ways so has double the parameters of a regular LSTM layer
outputs = layers.Dense(6, activation="softmax")(x)
model_4 = tf.keras.Model(inputs, outputs, name="model_4_Bidirectional")

# Compile
model_4.compile(loss="CategoricalCrossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

# Get a summary of our bidirectional model
model_4.summary()

# Fit the model (takes longer because of the bidirectional layers)
model_4_history = model_4.fit(train_sentences,
                              train_labels,
                              epochs=4,
                              validation_data=(val_sentences, val_labels),
                              callbacks=[create_tensorboard_callback(SAVE_DIR, "bidirectional_RNN")])

# Make predictions with bidirectional RNN on the validation data
model_4_pred_probs = model_4.predict(val_sentences)
model_4_pred_probs[:10]

# Convert prediction probabilities to labels
model_4_preds = tf.squeeze(tf.round(model_4_pred_probs))
model_4_preds[:10]

# Calculate bidirectional RNN model results
model_4_results = calculate_results(val_labels, model_4_preds)
model_4_results

# model_1_results

# model_2_results

# model_3_results

model_4_results

# sen=['what the hell is this','wow amazing this good','pls dont hurt me i dont like it']
# guess=model_4.predict(sen)
# guess

# result=tf.squeeze(tf.round(guess))

# print(result)

# 2:anger, 9:disappointment, 14:fear, 17:joy, 26:surprise, 27:neutral
# sentence=input("Please enter a sentence: ")
def translate_text(text, target_language='en'):
    # Initialize the translator
    translator = Translator()

    # Translate the text to the target language
    translation = translator.translate(text, dest=target_language)

    return translation.text


def analysis(sentence):
   print("Sentence has been passed")
   print(sentence)
   translated_input= translate_text(sentence, target_language='en')
   answer= model_4.predict([translated_input])
   result=tf.squeeze(tf.round(answer))
   
   first_element = result[0]
   second_element = result[1]
   third_element = result[2]
   fourth_element = result[3]
   fifth_element = result[4]
   sixth_element = result[5]
   

   if(first_element==1 and second_element==0 and third_element==0 and fourth_element==0 and fifth_element==0 and sixth_element==0):
      return("Fear")
   elif(first_element==0 and second_element==1 and third_element==0 and fourth_element==0 and fifth_element==0 and sixth_element==0):
      return("Joy")
   elif(first_element==0 and second_element==0 and third_element==1 and fourth_element==0 and fifth_element==0 and sixth_element==0):
      return("Angry")
   elif(first_element==0 and second_element==0 and third_element==0 and fourth_element==1 and fifth_element==0 and sixth_element==0):
      return("Surprise")
   elif(first_element==0 and second_element==0 and third_element==0 and fourth_element==0 and fifth_element==1 and sixth_element==0):
      return("Neutral")
   else:return("Disappointed")

# answer1251209364 = analysis([sentence])
# print("answer: " + answer1251209364)
