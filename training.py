import emot
import os
import matplotlib.pyplot as plt
import pandas as pd
from gensim.models import FastText 
import re
import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense,Concatenate,TimeDistributed,Masking,GRU,Input,Dot,Reshape,Softmax
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import numpy as np
from gensim.models import FastText 
from random import seed
from random import randint
from sklearn.cluster import KMeans
from config import *
from model import model_
from config import *
from preprocess import *


def training(WEIGHTS_PATH,CHECKPOINTS_PATH,dataset,model_,lr=lr,iterations=iterations,return_bestweightspath=return_bestweightspath):
  abae=model_(embed_outputdim,aspects_k)
  optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
  @tf.function
  def train_step(input):
      with tf.GradientTape() as tape:
        loss = abae(input)[0]
      gradients = tape.gradient(loss, abae.trainable_variables)
      optimizer.apply_gradients(zip(gradients, abae.trainable_variables))
      return loss, gradients
  train_loss = tf.keras.metrics.Mean(name='train_loss')

  
##check point to save
  ckpt = tf.train.Checkpoint(optimizer=optimizer, model=abae)
  ckpt_manager = tf.train.CheckpointManager(ckpt, CHECKPOINTS_PATH, max_to_keep=3)
  loss_list=[]
  weights_pathlist=[]
  for k in range(0,iterations): # k - number of iterations
    counter = 0
  
  # navigating through each batch
    for input in dataset:
      loss_, gradients = train_step(input)
      #adding loss to train loss
      train_loss(loss_)
      counter = counter + 1
      template = '''Done {} step, Loss: {:0.6f}'''
      if counter%500==0:
        print(template.format(counter, train_loss.result()))
#
    loss_list.append(train_loss.result()) #appending loss after every epoch
    ckpt_save_path  = ckpt_manager.save() #checkpointing after every epoch
    if os.path.isdir(WEIGHTS_PATH)==False:
      os.makedirs(WEIGHTS_PATH)
    X=os.path.join(WEIGHTS_PATH,"weights_epoch_"+str(k+1))
    abae.save_weights(X,save_format="h5")

    weights_pathlist.append(X)
    
    print("weights saved after epoch {}".format(k+1))
    print ('Saving checkpoint for iteration {} at {}'.format(k+1, ckpt_save_path))
    print(counter, train_loss.result())
    train_loss.reset_states()             #resetting loss after every epoch
  if return_bestweightspath:
    argminimum=np.argmin(loss_list)
    return weights_pathlist[argminimum]  


if __name__=="__main__":
  dataset=generate_dataset(buffer_size=buffer_size,batch_size=batch_size,negative_samples=negative_samples)
  weight_path=training(WEIGHTS_PATH,CHECKPOINTS_PATH,dataset,model_,lr=lr,iterations=iterations,return_bestweightspath=return_bestweightspath)



  
  