import numpy as np
import nltk 

from nltk.stem.porter import PorterStemmer
stemmer= PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokennize_sentence,all_words):
    
    tokennize_sentence = [stem(w)for w in tokennize_sentence]
    bag = np.zeros(len(all_words), dtype = np.float32)

    for idx, w in enumerate(all_words):
        if w in tokennize_sentence:
            bag[idx]=1.0
    return bag

