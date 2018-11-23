import pickle
import pandas as pd
import csv
import spacy
import argparse

class Preprocess():
    # Parse the command line arguments
    def __init__(self):
        parser = argparse.ArgumentParser(description='preprocess.py')
        
        parser.add_argument('-src', required=True, help='Path to the source data text file')
        parser.add_argument('-tgt', required=True, help='Path to the target data text file')
        parser.add_argument('-glove', required=True, help='Path to GLoVE embedding text file')
        parser.add_argument('-save', required=True, help='Save path for output')
        
        self.options = parser.parse_args()

    # Load the souce and target text files. Lowercase everything
    def load_data(self):
        with open(self.options.src, 'r') as f:
            self.sentences = [line.rstrip('\n').lower() for line in f]
        with open(self.options.tgt, 'r') as f:
            self.questions = [line.rstrip('\n').lower() for line in f]

    # Tokenize source and target, save as pickel files in the save path
    def tokenize(self):
        tok = spacy.load('en')
        src_toks = [[t.text for t in tok.tokenizer(s)] for s in self.sentences]
        tgt_toks = [[t.text for t in tok.tokenizer(q)] for q in self.questions]

        with open(self.options.save+'src_tokenized.pkl', 'wb') as fp:
            pickle.dump(src_toks, fp)
        with open(self.options.save+'tgt_tokenized.pkl', 'wb') as fp:
            pickle.dump(tgt_toks, fp)

    # Load the word embeddings. Create a dictionary from word to vector.
    # Save as pickle
    def word_emb(self):
        words = pd.read_table(self.options.glove, sep=" ", index_col=0, header=None, \
                                        quoting=csv.QUOTE_NONE)
        glove = {key: val.values for key, val in words.T.items()}
        with open(self.options.save+'glove.840B.300d.pkl', 'wb') as fp:
            pickle.dump(glove, fp)


if __name__ == "__main__":
    preproc = Preprocess()
    preproc.load_data()
    preproc.tokenize()
    preproc.word_emb()
