import pandas as pd
import numpy as numpy
import sys
from nltk.tokenize import RegexpTokenizer

inputfile = sys.argv[1]
df = pd.read_csv(inputfile)

tokenizer = RegexpTokenizer(r'\w+')
df['tokens'] = df['text'].apply(tokenizer.tokenize)
print(df)

MAX_SEQUENCE_LENGTH = 0
flag = True
for tokens in df['tokens']:
	MAX_SEQUENCE_LENGTH = max(MAX_SEQUENCE_LENGTH, len(tokens))
	if MAX_SEQUENCE_LENGTH == 319 and flag:
		print(tokens)
		flag = False

print(MAX_SEQUENCE_LENGTH)
