import numpy as np
import pandas as pd
import sys
import os
import re
from scipy import spatial
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from typing import List
from numpy.linalg import norm
from tqdm import tqdm

sys.path.insert(
    0, os.path.join(os.getcwd(), "Learning_equality_curriculum_recommendation/logs")
)

sys.path.insert(
    0, os.path.join(os.getcwd(), "Learning_equality_curriculum_recommendation/configs")
)

from Learning_equality_curriculum_recommendation.configs.confs import (
    load_conf,
    clean_params,
    Loader,
)

main_params = load_conf("configs/main.yml", include=True)
main_params = clean_params(main_params)

sentence_bert_type = main_params["model_type"]
embedded_data_path = main_params["embedded_data_path"]

embedded_description = pd.read_csv(
    os.path.join(embedded_data_path, "description_embdedd.csv")
).dropna()
embedded_text = pd.read_csv(
    os.path.join(embedded_data_path, "df_text_embedded.csv")
).dropna()
embedded_title = pd.read_csv(
    os.path.join(embedded_data_path, "df_title_embedded.csv")
).dropna()
full_embedded = embedded_description.merge(embedded_text, on="id", how="left")
full_embedded.dropna(inplace=True)
topics = pd.read_csv("data/topics.csv")

from logs import *

tqdm.pandas()
main()


def global_clean(string_list: str) -> List[int]:
    """
    The goal of this function is to clean all
    embedded values from string lists to normal
    lists

    Arguments:
        -string_list: str: The list represented
        as a string

    Returns:
        -cleaned_list: list: The list after it
        was cleaned and converted
    """
    string_list = re.sub(r"\s+", ",", string_list)

    if "[," in string_list:
        string_list.replace("[,", "[")
    try:
        cleaned_list = eval(string_list)
        return cleaned_list
    except SyntaxError:
        cleaned_list = eval(string_list.replace("[,", "["))
        return cleaned_list


logging.info("Cleaning all embeddings...")
full_embedded["description"] = full_embedded["description"].progress_apply(
    lambda x: global_clean(x)
)
full_embedded["text"] = full_embedded["text"].progress_apply(lambda x: global_clean(x))
logging.info("Cleaning successfully acheived !")


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
        self.model = SentenceTransformer(sentence_bert_type)
        logging.info("The Sentence Bert Model has been initialized")

    def get_embedding(self, column: str) -> pd.DataFrame():
        """
        The goal of this function is to embedd the different
        textual data thanks to the Sentence Bert model

        Arguments:
            -column: str: The column to be embedded

        Returns:
            -self.df: pd.DataFrame: The DataFrame with
            the embedded column
        """

        if "embedded" + column in self.df.columns:
            return self.df

        logging.info(f"The embedding of the {column} column has begun")
        self.df["embedded" + column] = self.df[column].progress_apply(
            lambda x: self.model.encode(x)
        )

        return self.df

    def get_single_correlation(
        self,
        topic_id: str,
        min_correlation: float,
        column_compared: str,
        dataframe_compared=full_embedded,
    ) -> List[str]:
        """
        The goal of this function is to return, for each sentence,
        the most similar other sentences

        Arguments:
            -topic_id: str: The topic id as it is reported in
            the DataFrame
            -min_correlation: int: The minimum correlation required
            with the input sentence to be retained
            -column_compared: str: The column to which the topic id will
            be compared to get the most similar content
            -dataframe_compared: pd.DataFrame: The DataFrame embedded
            values will be taken from
        Returns:
            top_correlated: List[str]: The
        """

        assert column_compared in [
            "text",
            "title",
            "description",
        ], "The model can only compare\
the ids with the columns text, title and description"

        inp_text = topics.loc[topics.id == topic_id, column_compared].values[0]
        inp_vector = self.model.encode(inp_text)
        s = dataframe_compared[column_compared].progress_apply(
            lambda x: 1 - spatial.distance.cosine(x, inp_vector)
        )
        dataframe_compared = dataframe_compared.assign(similarity=s)
        top_correlated = (
            dataframe_compared[dataframe_compared.similarity >= min_correlation]
            .sort_values("similarity", ascending=False)["id"]
            .tolist()
        )

        return top_correlated


def cosine(x: np.array, y: np.array) -> float:
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

    similarity = np.dot(x, y) / (norm(x) * norm(y))
    return similarity


def get_all_cosine(index: int, df: pd.DataFrame, column: str) -> np.array:
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

    logging.info(f"The cosine similarity computation has begun")
    df_extract = df.drop(index, axis=0)
    candidates = df_extract[column].tolist()
    reference = df.loc[i, column]
    similarities = [cosine(reference, x) for x in candidates]

    return similarities
