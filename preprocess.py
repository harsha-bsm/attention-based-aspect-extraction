import emot
import os
import io
import matplotlib.pyplot as plt
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
nltk.download('stopwords')
from nltk.corpus import stopwords
import numpy as np
from gensim.models import FastText 
from random import seed
from random import randint
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
from config import *

#  removing some of the common contractions.
def decontractions(phrase):
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"won\’t", "will not", phrase)
    phrase = re.sub(r"can\’t", "can not", phrase)
    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    phrase = re.sub(r"n\’t", " not", phrase)
    phrase = re.sub(r"\’re", " are", phrase)
    phrase = re.sub(r"\’s", " is", phrase)
    phrase = re.sub(r"\’d", " would", phrase)
    phrase = re.sub(r"\’ll", " will", phrase)
    phrase = re.sub(r"\’ve", " have", phrase)
    phrase = re.sub(r"\’m", " am", phrase)
    return phrase

def convert_emojis(review):
  for x in review:
    if x in emot.emo_unicode.UNICODE_EMOJI.keys():
      review=review.replace(x,"")
  return review

#Converting all the extra spaces into single space. This will help while splitting the data in the future.
def removecharacters(review):
  review= re.sub('[^A-Za-z0-9]+',' ',review)  #anything exept numbers and alphabets, replace them with space
  review=re.sub(r"\n"," ",review)  #new lines into space
  review=re.sub(r"\t"," ",review)  #tabs into space
  review=re.sub(r"\v"," ",review)  #vertical tab into space
  review=re.sub(r"\s"," ",review)   #all extra spaces into single space
  return review.lower()
  
def removenewords(review,wordcountdict):
  return " ".join([word for word in review.split(" ") if  word in list(wordcountdict.keys())])

def removestopwords(review,stopword):
  return " ".join([word for word in review.split(" ") if not word in stopword])

def word_count(review,wordcountdict,min_word_repeat):
  return  " ".join([word for word in review.split() if wordcountdict[word]>min_word_repeat] )



def text_processing(stopword=stopword,min_word_repeat=10,sent_len_percentile=99.9,dump_aspickle=True,return_df=False):
  #stop_words=set(stopwords.words('english'))
  raw_path=os.path.join(DATA_PATH,raw_file)
  preprocessed_path=os.path.join(DATA_PATH,preprocessed_file)
  if os.path.isfile(preprocessed_path):
    preprocessed_df=pd.read_pickle(preprocessed_path)
  else:
    df = pd.read_pickle(raw_path)
    df["review"]=df["review"].apply(lambda x:x.strip("READ MORE").lower())
    df["review"]=df["review"].apply(convert_emojis)
    df["review"]=df["review"].apply(decontractions)
    df["review"]=df["review"].apply(removecharacters)
    df["review"]=pd.DataFrame(df["review"].apply(lambda x:removestopwords(x,stopword)))
    df["len"]=df.review.str.split().apply(len)
    l=np.percentile(df.len,sent_len_percentile)
    df=df.loc[(df["len"]>0) & (df["len"]<l+1)]
    tkn = Tokenizer(filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n') #tensorflow tokenising
    tkn.fit_on_texts(df['review'].values)
    word_countdict=tkn.word_counts  #word count dictionary
    df["review"]=pd.DataFrame(df["review"].apply(lambda x:word_count(x,word_countdict,min_word_repeat)))
    df["len"]=df.review.str.split(" ").apply(len)
    if  dump_aspickle:
      with open(preprocessed_path,"wb") as file:
        pickle.dump(df.loc[df["len"]>0],file)
    preprocessed_df=df.loc[df["len"]>0]
  if return_df:
    return preprocessed_df


def tokenisation_on_traindata(return_tkn=True):
  if os.path.isfile(os.path.join(DATA_PATH,preprocessed_file)):
    train=pd.read_pickle(os.path.join(DATA_PATH,preprocessed_file))
  else:
    train=text_processing(stopword=stopword,min_word_repeat=10,sent_len_percentile=99.9,dump_aspickle=True,return_df=True)
  tkn = Tokenizer(filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n') #tensorflow tokenising
  tkn.fit_on_texts(train['review'].values)
  with io.open(os.path.join(DATA_PATH,"token.pickle"),"wb") as file:
    pickle.dump(tkn,file)
  if return_tkn:
    return tkn


def training_vocab(embed_dim=100,negative_sampling=5,min_count=1,window=10,iter=250,sg=1,train_again=vocabtrain_again,return_model=True):
  preprocessed_path=os.path.join(DATA_PATH,preprocessed_file)
  if os.path.isfile(preprocessed_path):
    df=pd.read_pickle(preprocessed_path)
  else:
    df=text_processing(stopword=set(stopwords.words('english')),return_df=True)
  tkn = Tokenizer(filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n')
  tkn.fit_on_texts(df['review'].values)
  word_countdict=tkn.word_counts
  seq_texts=tkn.texts_to_sequences(df['review'].values)  #converting tokenised reviews into sequence of
# integers with each integer corresponding to one word in the corpus
  text=pd.Series(df['review'].values).apply(lambda x: x.split())  #spliting the reviews in a format suitable to  feed 
#into Fasttext for trainig vocab 

  trained_embeddings_path=os.path.join(DATA_PATH,trained_embeddings)
  if os.path.isfile(trained_embeddings_path):
    if train_again:
      model = FastText(size=embed_dim, negative=negative_sampling, min_count=min_count,window=window,iter=iter,sg=sg) 
      model.build_vocab(text) 
      gensim_fasttext = model.train(sentences=text, 
                           sg=sg, ##skipgram
                           epochs=iter, ##no of iterations
                           size=embed_dim, ##dimentions of word embedding
                           seed=1,
                           total_examples=model.corpus_count)
      model.save(trained_embeddings_path)
    else:
      model=FastText.load(trained_embeddings_path)
  else:
    model = FastText(size=embed_dim, negative=negative_sampling, min_count=min_count,window=window,iter=iter,sg=sg) 
    model.build_vocab(text)
    gensim_fasttext = model.train(sentences=text, 
                           sg=sg, ##skipgram
                           epochs=iter, ##no of iterations
                           size=embed_dim, ##dimentions of word embedding
                           seed=1,
                           total_examples=model.corpus_count)
    model.save(trained_embeddings_path)
  
  
  if return_model:
    return model

def textinputsequence_padding(padding="post",padded_seqagain=train_again,return_paddedsequences=True):
  preprocessed_path=os.path.join(DATA_PATH,preprocessed_file)
  padded_seqpath=os.path.join(DATA_PATH,padded_seqfile)
  if os.path.isfile(preprocessed_path):
    df=pd.read_pickle(preprocessed_path)
  else:
    df=text_processing(stopword=set(stopwords.words('english')),return_df=True)
  if os.path.isfile(padded_seqpath):

    if padded_seqagain:
      tkn = Tokenizer(filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n')
      tkn.fit_on_texts(df['review'].values)
      word_countdict=tkn.word_counts
      seq_texts=tkn.texts_to_sequences(df['review'].values)
      seq_texts=tf.keras.preprocessing.sequence.pad_sequences(seq_texts,
                                                         maxlen=max(df.len),
                                                         padding=padding)
      with open(os.path.join(DATA_PATH,"padded_sequences.pickle"),"wb") as file:
        pickle.dump(seq_texts,file)
    else:
      with open(padded_seqpath,"rb") as file:
        seq_texts=pickle.load(file)
  else:
    tkn = Tokenizer(filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n')
    tkn.fit_on_texts(df['review'].values)
    word_countdict=tkn.word_counts
    seq_texts=tkn.texts_to_sequences(df['review'].values)
    seq_texts=tf.keras.preprocessing.sequence.pad_sequences(seq_texts,
                                                         maxlen=max(df.len),
                                                         padding=padding)
    with open(os.path.join(DATA_PATH,"padded_sequences.pickle"),"wb") as file:
      pickle.dump(seq_texts,file)
    with open(os.path.join(DATA_PATH,"maxlen.pickle"),"wb") as file:
      pickle.dump(seq_texts.shape[1],file)

  if return_paddedsequences:
    return seq_texts


def generate_dataset(buffer_size=buffer_size,batch_size=batch_size,negative_samples=negative_samples):
  if  os.path.isfile(os.path.join(DATA_PATH,padded_seqfile)):
    with io.open(os.path.join(DATA_PATH,"padded_sequences.pickle"),"rb") as file:
      seq_texts=pickle.load(file)
  else:
    seq_texts=textinputsequence_padding(padding="post",padded_seqagain=train_again,return_paddedsequences=True)

  def gendata():
    seed(42)
    for i in range(0,len(seq_texts)):
      lis=[]
      lent=[]
      while len(lent)<negative_samples:
        value = randint(0, len(seq_texts)-1)
        if value==i:
          continue
        lis.append(seq_texts[value])
        lent.append(value)
      yield seq_texts[i],lis
  dataset=tf.data.Dataset.from_generator(gendata, output_types=(tf.int32,tf.int32))
  dataset=dataset.repeat(1).shuffle(buffer_size=buffer_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
  return dataset

if __name__=="__main__":
  dataset=generate_dataset(buffer_size=buffer_size,batch_size=batch_size,negative_samples=negative_samples)
  tf.data.experimental.save(
    dataset, path="/content/drive/My Drive/REVIEWSV2/", compression=None, shard_func=None)

