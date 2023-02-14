# The goal of this python file is to create an embedding model
# taking into account the different languages
import pandas as pd


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


class Embedding(Preprocessor):
    """
    The goal of this class is to inherit from a preprocessed
    DataFrame with textual information ready to be embedded
    """

    def __init__(self, df) -> None:
        self.df = df
