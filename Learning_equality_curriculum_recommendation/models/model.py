import numpy as np 
import pandas as pd
import sys 
import os 
from scipy import spatial
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from typing import List
from numpy.linalg import norm
from tqdm import tqdm 

sys.path.insert(0, os.path.join(os.getcwd(),"Learning_equality_curriculum_recommendation/logs"))

from logs import *

tqdm.pandas()
main()

class Sentence_Bert_Model:
    """
    The goal of this class is to use the Sentence
    Bert Model as a measure of the similarity between 
    two given sequences

    Arguments:
        -df: pd.DataFrame: The DataFrame on which the
        model will be applied 
    """
    
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df
        self.model = SentenceTransformer('distiluse-base-multilingual-cased')
        logging.info("The Sentence Bert Model has been initialized")

    def get_embedding(self, column: str)->pd.DataFrame():
        """
        The goal of this function is to embedd the different 
        textual data thanks to the Sentence Bert model
        
        Arguments:
            -column: str: The column to be embedded
            
        Returns: 
            -self.df: pd.DataFrame: The DataFrame with 
            the embedded column
        """

        
        if "embedded"+column in self.df.columns:
            return self.df
        
        logging.info(f"The embedding of the {column} column has begun")
        self.df["embedded"+column]=self.df[column].progress_apply(lambda x: self.model.encode(x))

        return self.df

    def get_single_correlation(self, input_sentence: str, limit: int, column: str)->List[str]:
        """
        The goal of this function is to return, for each sentence,
        the most similar other sentences 

        Arguments:
            -input_sentence: str: The sentence which most similar 
            ones we are looking for 
            -limit: int: The upper number of similar sentences we 
            want
            -column: str: The column we are working on and 
            in which we want to find similar

        Returns:
            top_correlated: List[]
        """
        data=self.df.copy()
        inp_vector=self.model.encode(input_sentence)
        data[column]=data[column].progress_apply(lambda x: self.model.encode(x))
        s=data[column].progress_apply(lambda x: 1 - spatial.distance.cosine(x, inp_vector) )
        data=data.assign(similarity=s)
        top_correlated=data.sort_values('similarity',ascending=False).head(limit)["id"].tolist()    

        return top_correlated

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

def get_all_cosine(index: int, df: pd.DataFrame, column: str)->np.array:
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
    