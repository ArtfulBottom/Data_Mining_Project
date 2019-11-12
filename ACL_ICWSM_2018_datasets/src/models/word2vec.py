import pandas as pd
import numpy as numpy
import sys
from nltk.tokenize import RegexpTokenizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

# Read input data.
inputfile = sys.argv[1]
df = pd.read_csv(inputfile)

# Tokenize data.
tokenizer = RegexpTokenizer(r'\w+')
df['tokens'] = df['text'].apply(tokenizer.tokenize)

# Compute maximum token length of tweet.
MAX_SEQUENCE_LENGTH = 0
for tokens in df['tokens']:
	MAX_SEQUENCE_LENGTH = max(MAX_SEQUENCE_LENGTH, len(tokens))

all_words = [word for tokens in df['tokens'] for word in tokens]
VOCAB = sorted(list(set(all_words)))
VOCAB_SIZE = len(VOCAB)

# Pad data.
tokenizer = Tokenizer(num_words=MAX_SEQUENCE_LENGTH)
tokenizer.fit_on_texts(df['text'].tolist())
sequences = tokenizer.texts_to_sequences(df['text'].tolist())
print(df['text'])
print(len(sequences))
print(sequences[0])
