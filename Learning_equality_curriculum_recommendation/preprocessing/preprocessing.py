# The goal of this python file is to create an embedding model
# taking into account the different languages
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from tqdm import tqdm

tqdm.pandas()


class Preprocessor:
    """
    The goal of this class is to extract the right textual
    information from the DataFrame and having them ready for
    the embedding
    """

    def __init__(self, df: pd.DataFrame()) -> None:
        """
        The goal of this function is to init the parameters
        of the class

        Arguments:
            -df: pd.DataFrame(): The DataFrame with the textual
            data to be processed

        Returns:
            -None
        """
        self.df = df
        self.target_columns = ["title", "description", "text"]

    def cleaning_missing_values(self) -> pd.DataFrame():
        """
        The goal of this function is to remove the missing
        values from the DataFrame to have usable data

        Arguments:
            None

        Returns:
            -df: pd.DataFrame: The DataFrame that has just
            been treated
        """
        self.df.title.fillna("-", inplace=True)
        self.df.description.fillna("-", inplace=True)
        self.df.text.fillna("-", inplace=True)

        return self.df

    def tokenization(self) -> pd.DataFrame():
        """
        The goal of this function is creating tokens
        from the DataFrame to have tokenized textual
        data

        Arguments:
            -None

        Returns:
            -df: pd.DataFrame: The DataFrame that has just
            been treated
        """
        self.df.title = self.df.title.progress_apply(lambda x: word_tokenize(x))
        self.df.description = self.df.description.progress_apply(
            lambda x: word_tokenize(x)
        )
        #self.df.text = self.df.text.progress_apply(lambda x: word_tokenize(x))

        return self.df

    def removing_stopwords(self) -> pd.DataFrame():
        """
        The goal of this function is removing stopword
        (words useless for the context) in the DataFrame
        textual data

        Arguments:
            -None

        Returns:
            -df: pd.DataFrame: The DataFrame that has just
            been treated
        """
        pass


class Embedding(Preprocessor):
    """
    The goal of this class is to inherit from a preprocessed
    DataFrame with textual information ready to be embedded
    """

    def __init__(self, df) -> None:
        self.df = df
