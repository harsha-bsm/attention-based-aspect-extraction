import os
import pandas as pd
from gensim.models import FastText 
import re
import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense,Concatenate,TimeDistributed,Masking,GRU,Input,Dot,Reshape,Softmax
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
# Using NLTK for PoS tagging and Stopwords removal
# Downloading the corresponding packages
import pickle
import nltk
from nltk.corpus import stopwords
import numpy as np
from gensim.models import FastText 
from random import seed
from random import randint
from sklearn.cluster import KMeans
from config import *
from preprocess import *

def embedding_layerinit():
  model=training_vocab(embed_dim=embed_outputdim,negative_sampling=5,min_count=1,window=10,iter=250,sg=1,train_again=vocabtrain_again,return_model=True)
  trained_weights=np.vstack((np.zeros((1,100)),model.wv.vectors))
  return trained_weights,model

def inputlength():
  seq=textinputsequence_padding(padding="post",padded_seqagain=train_again,return_paddedsequences=True)
  return seq.shape[1]


def aspectlayer_weightinit():
  trained_weights=embedding_layerinit()[0]
  kmeans = KMeans(n_clusters=aspects_k,random_state=0,max_iter=500,n_jobs=-1).fit(trained_weights) #clustering the trained wordembeddings into 10 clusters
  init=tf.constant_initializer(kmeans.cluster_centers_) #used for initialing the weights of final dense layer
  return init

class Attention(tf.keras.layers.Layer):
    def __init__(self):
      super().__init__()
      #self.embed_outputdim=embed_outputdim
      self.soft = tf.keras.layers.Softmax(axis=-2,name="softmax_att") #softmax layer
      self.dot=tf.keras.layers.Dot(axes=(-1,-1),name="dot_att")       #dot layer
      self.w = tf.Variable(
            initial_value=tf.random_normal_initializer()(shape=(embed_outputdim,embed_outputdim), dtype="float32"),
            trainable=True,
        )              #weights that captures essense between the word embedding and global context vector(or average of all the word embedding of the sentence)
  
    def call(self,embed_output, mask=None):

      ys = tf.reduce_mean(embed_output,axis=-2) #average of all word embeddings of the sentence
      ys=tf.expand_dims(ys,axis=-2)
      eW=tf.matmul(embed_output, self.w)
      eW = eW * tf.expand_dims(tf.cast(mask,tf.float32),-1) #maskpropagation. Preventing masked elements into calculations
      f=self.dot([eW, ys])
      f = f+tf.expand_dims(tf.cast(tf.math.equal(mask, False), f.dtype)*-1e9,-1) #multiplying all the masked elements by -1e9 so that the softmax step do not impact the vectors
      f=self.soft(f)
      zs=tf.math.reduce_sum(f*embed_output,axis=-2) #zs is aspect embedding space after attention mechanisms on words of the sentence
                                                    # f - softmax - gives info about unimportant words in the sentence for extracting aspects 
      
      return zs,f
    @classmethod
    def from_config(cls, config):
      return cls(**config)


# custom model

class model_(tf.keras.Model):
  def __init__(self,embed_outputdim,aspects_k):
    super().__init__()
    self.embed_inputdim=len(embedding_layerinit()[1].wv.vocab)
    self.inputlength=inputlength()
    self.embed_outputdim=embed_outputdim
    self.trained_weights=embedding_layerinit()[0]
    self.embedding=Embedding(input_dim=self.embed_inputdim+1,output_dim=self.embed_outputdim,mask_zero=True,
                             input_length=self.inputlength,weights=[self.trained_weights],name="embedding_layer",trainable=True) #embedding layer. Zero masking
    self.attention=Attention()
    self.aspects_k=aspects_k #number of aspects
    self.init=aspectlayer_weightinit()
    self.k = tf.keras.layers.Dense(aspects_k,name="dim_reduction_layer",activation="sigmoid")
    self.dense= self.trained_weights.shape[1]
    self.final=tf.keras.layers.Dense(self.dense,name="final_dense",kernel_initializer=self.init) #weights are initialised with embedding clusters
  def call(self,input):
    e=self.embedding(input[0])
    mask = self.embedding.compute_mask(input[0]) #computing mask so that this this can be used while propagating mask for subsequent layers
    zs=self.attention(e, mask = mask)[0] #aspect vector
    pt=self.k(zs)  #dimensionality vector
    rs=self.final(pt) #reconstructed vector

    # building regulariser to be used in the loss. [(T*transpose(T))-I]
    reg=tf.tensordot(tf.linalg.normalize(self.final.weights[0],axis=1)[1],tf.linalg.normalize(self.final.weights[0],axis=1)[1],axes=[[1],[1]])
    reg=reg-tf.ones([self.aspects_k,self.aspects_k])
    reg=tf.norm(reg, ord='euclidean', axis=None, keepdims=None, name=None)
    # calculating loss
    r=tf.expand_dims(rs,-2)
    f=tf.tensordot(rs,zs,[[0,1],[0,1]])
    a=self.embedding(input[1])
    a=tf.reduce_mean(a,axis=-2)
    loss=tf.reduce_sum(tf.nn.relu(1-tf.reduce_sum(tf.tensordot(a,r,[[0,2],[0,2]]))+f))+lamda*reg #this is the loss which is to be minimised
    return loss,pt,rs 
    
    
  def get_config(self):
    config = {
                  'embed_outputdim': self.embed_outputdim,
                  'aspects_k' : self.aspects_k}
    return config
  @classmethod
  def from_config(cls, config):
    return cls(**config)