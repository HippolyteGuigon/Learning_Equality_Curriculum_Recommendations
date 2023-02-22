from sklearn.metrics.pairwise import cosine_similarity
import numpy as np 
import pandas as pd
from numpy.linalg import norm

def cosine(x: np.array, y: np.array)->float:
    """
    The goal of this function is to compute the 
    cosine similarity between two vectors to determine
    how close they are 
    
    Arguments:
        -x: np.array(): The first vector to be considered
        -y: np.array(): The second vector to be considered
    
    Returns:
        -similarity: float: The similarity computed between 
        the two vectors 
    """

    similarity=np.dot(x,y)/(norm(x)*norm(y))
    return similarity

def get_all_cosine(index: int, df: pd.DataFrame(), column: str)->np.array():
    """
    The goal of this function is to compute all the cosine similarities 
    between one vector and all the others in the DataFrame
    
    Arguments:
        -index : int: The index from which the candidate in the DataFrame 
        will be extracted
        -df: pd.DataFrame: The DataFrame from which all candidates will be 
        extracted 
        -column: str: The column name in which the similarity computation 
        will take place
    """
    df_extract=df.drop(index, axis=0)
    candidates=df_extract[column].tolist()
    reference=df.loc[i, column]
    similarities=[cosine(reference, x)  for x in candidates]
    
    return similarities
    