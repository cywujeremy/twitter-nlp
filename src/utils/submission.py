import pandas as pd
import numpy as np

def prediction_output(series, file, id=None):
    """output a prediction series to csv file
    Args:
        series (numpy.array): a series of prediction
        file (string): output file path
        id (numpy.array): specify the output indexes for the observations
    """
    if id is None:
        id = np.arange(len(series))

    df = pd.DataFrame({'id': id,
                       'target': series})

    df.to_csv(file, index=False)