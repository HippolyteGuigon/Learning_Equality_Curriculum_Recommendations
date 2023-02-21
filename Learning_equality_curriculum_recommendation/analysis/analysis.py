# The goal of this python file is to perform unsupervised analysis
# on data that has just been preprocessed

import texthero as hero
import pandas as pd


class Dimension_Reduction:
    """
    The goal of this class is to apply
    dimension reduction techniques on
    the preprocessed data in order to have
    insights on its internal structure

    Arguments:
        -df: pd.DataFrame: The Dataframe
        on which unsupervised methods will
        be applied
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    def pca(self) -> pd.DataFrame:
        """
        The goal of this function is applying the
        Principal Component analysis method on the
        preprocessed DataFrame to gain insight on
        its data in lower dimension

        Arguments:
            -None

        Returns:
            -self.df: pd.DataFrame: The DataFrame after
            dimension reduction was applied
        """

        self.df["clean_title_pca"] = hero.do_pca(self.df["title"])

        return self.df

    def visualize_pca(self, column: str = "language") -> None:
        """
        The goal of this function is to visualize PCA
        once it has been applied

        Arguments:
            -column: str: The column which will be used
            as colour

        Returns:
            -None
        """
        hero.scatterplot(
            self.df,
            col="clean_title_pca",
            color=column,
            title="PCA  visualisation of language distribution",
        )
