# -*- coding: utf-8 -*-

#Importing libraries and importing data

import tensorflow as tf
print(tf.__version__)

# Here we import everything we need for the project

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
# import matplotlib.pyplot as plt
#import cv2
import pandas as pd


#Importing translator library
from googletrans import Translator

# Sklearn
from sklearn.model_selection import train_test_split # Helps with organizing data for training
# from sklearn.metrics import confusion_matrix # Helps present results as a confusion-matrix

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


# They can have any number of arguments but only one expression.

# Before we start to further process the data, we will first have to combine all the three datasets into one and then proceed further


# Combining all the 3 datasets into 1 dataset
df_all = pd.concat([df_train, df_val, df_test], axis=0).reset_index(drop=True)

# Preview of data
print(df_all.head(3))

print(df_all.shape)

"""Next, we will have to map each GE_indices to it's corresponding emotion in the emotions.txt file

For that, let's first convert these GE_indices into a list of indices which we will then map to the emotions.txt file and extract the emotion labels from the file
"""


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




import wget
url="https://raw.githubusercontent.com/mrdbourke/tensorflow-deep-learning/main/extras/helper_functions.py"
filename=wget.download(url)


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
