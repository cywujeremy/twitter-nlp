import numpy as np
import pandas as pd
import re

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
