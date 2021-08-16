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
from nltk.corpus import stopwords
import numpy as np
from gensim.models import FastText 
from random import seed
from random import randint
from sklearn.cluster import KMeans
from config import *
from model import model_
from preprocess import *
from training import *



def seq_gen(testsample,maxlen=223)  :
  df=pd.DataFrame([testsample],columns=["review"])
  df["review"]=df["review"].apply(lambda x:x.strip("READ MORE").lower())
  df["review"]=df["review"].apply(convert_emojis)
  df["review"]=df["review"].apply(decontractions)
  df["review"]=df["review"].apply(removecharacters)
  df["review"]=pd.DataFrame(df["review"].apply(lambda x:removestopwords(x,stopword)))
  df["len"]=df.review.str.split().apply(len)
  df=df.loc[(df["len"]>0) & (df["len"]<maxlen+1)]
  tkn=tokenisation_on_traindata(return_tkn=True)
  word_countdict=tkn.word_counts  #word count dictionary
  df["review"]=pd.DataFrame(df["review"].apply(lambda x:removenewords(x,word_countdict)))
  df["review"]=pd.DataFrame(df["review"].apply(lambda x:word_count(x,word_countdict,10)))
  
  seq_texts=tkn.texts_to_sequences(df['review'].values)
  print(seq_texts)
  seq_texts=tf.keras.preprocessing.sequence.pad_sequences(seq_texts,
                                                         maxlen=maxlen,
                                                         padding='post')
  return seq_texts

def topicpredict(testsample,weight_path,model_,MODEL_CONFIG,maxlen=223):
  k=tf.random.uniform(shape=[100,223],minval=1,maxval=38,dtype=tf.int32),tf.random.uniform(shape=[100,20,223],minval=1,maxval=38,dtype=tf.int32)
  inf=model_.from_config(MODEL_CONFIG)
  inf(k)
  inf.load_weights(weight_path)
  l=inf.layers
  seq_text=seq_gen(testsample,maxlen=maxlen)
  sample=seq_text.reshape(1,maxlen)
  endsample=l[0](sample)
  mask=l[0].compute_mask(sample)
  att=l[1](endsample,mask)[0]  
  topic_num=np.argmax(l[2](att))+1
  
  return 


if __name__=="__main__":
  testsample=input()
  print(topicpredict(testsample,WEIGHTS_PATH,model_,MODEL_CONFIG,maxlen=223))
  
