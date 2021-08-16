from nltk.corpus import stopwords
DATA_PATH="/content/drive/My Drive/REVIEWS/"
raw_file="reviews.pickle"
preprocessed_file="preprocessed_df.pickle"
trained_embeddings="trained_embeddings"
padded_seqfile="padded_sequences.pickle"
train_again=True
vocabtrain_again=False
stopword=stopwords.words(fileids="english")
embed_outputdim=100
aspects_k=10 #number of aspects
buffer_size=1024
batch_size=100
negative_samples=20
#WEIGHTS_PATH="/content/drive/My Drive/REVIEWS/Weights"
WEIGHTS_PATH="/content/drive/My Drive/REVIEWS/trainedWEIGHTS"
CHECKPOINTS_PATH="/content/drive/My Drive/REVIEWS/checkpoints"
lr=0.001
iterations=15
return_bestweightspath=True

MODEL_CONFIG = {'embed_outputdim': embed_outputdim,
                  'aspects_k' : aspects_k}