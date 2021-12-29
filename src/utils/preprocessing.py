import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer

class TextNormalizer:
    """clean up text series using customized rules
    """

    def __init__(self, rules=None):
        self.rules = rules
    
    def clean(self, series):
        """clean a text series using the specified rules
        Args:
            series (numpy.array/list): a list of text
        Returns:
            series_clean (numpy.array): a list of cleaned text
        """

        series_clean = [doc.lower() for doc in series]

        for pattern, substitute in self.rules.items():

            series_clean = [re.sub(pattern, substitute, doc) for doc in series_clean]
        
        series_clean = np.array([doc.strip() for doc in series_clean])

        return series_clean


def generate_vocabulary(series, nlp_model):
    """generate and return both word-to-index dict and index-to-vector dict

    Args:
        series (pandas.Series): training text series
        nlp_model (spacy model): the vectorizing model to be used
    
    Returns:
        word2ind (dict): word-to-index dict
        ind2word (dict): index-to-word dict
        ind2vec (dict): index-to-vector dict
    """
    
    count_vectorizer = CountVectorizer()
    count_vectorizer.fit(series)
    
    words, _ = zip(*count_vectorizer.vocabulary_.items())
    words = list(words)
    words.insert(0, 'UKN')
    index = list(range(len(words)))
    
    word2ind = dict(zip(words, index))
    ind2word = dict(zip(index, words))
    
    vectors = [nlp_model(word).vector for word in words]
    vectors.insert(0, np.zeros(vectors[0].shape))
    ind2vec = dict(zip(index, vectors))
    
    return word2ind, ind2word, ind2vec


def encode_document(document, word2ind):
    """encode a string/document into a list of indexes using a specified dict

    Args:
        document (str): a string of document
        word2ind (dict): the encoding dictionary
    
    Returns:
        index_array (numpy.array): an array of indexes representing the words
    """
    
    index_array = np.array([word2ind[word] if word in word2ind.keys() else 0 for word in document.split()])
    
    return index_array