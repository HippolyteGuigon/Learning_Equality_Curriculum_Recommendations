# The goal of this python file is to perform unsupervised analysis
# on data that has just been preprocessed

#import texthero as hero
import pandas as pd
from typing import List
from tqdm import tqdm

tqdm.pandas()

def single_column_analysis(column_inspected: str)->pd.DataFrame:
    """
    The goal of this function is to get the kind (video, html) of given 
    content ids 
    
    Arguments:
        -column_inspected: str: The column that is to be 
        compared between topics and content
        
    Returns:
        -correlations: pd.DataFrame: The DataFrame with the compared
        column for each content id
    """
    correlations=pd.read_csv("data/correlations.csv")
    content=pd.read_csv("data/content.csv")
    topics=pd.read_csv("data/topics.csv")
    correlations["content_ids"]=correlations["content_ids"].apply(lambda x: x.split(" "))
    correlations=correlations.explode("content_ids")
    topics=topics[["id",column_inspected]].rename(columns={column_inspected:"topics_"+column_inspected})
    content=content[["id",column_inspected]].rename(columns={column_inspected:"content_"+column_inspected})
    correlations=correlations.merge(topics,left_on="topic_id",right_on="id",how="left")
    correlations=correlations.merge(content,left_on="content_ids",right_on="id",how="left")
    return correlations

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

    def visualize_pca(self, column="label") -> None:
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


class Clustering:
    """
    The goal of this class is to perform 
    unsupervised clustering methods on the 
    textual data that has just been preprocessed

    Arguments:
        -df: pd.DataFrame: The preprocessed DataFrame
        on which the clustering will be performed
    """

    def __init__(self, df:pd.DataFrame) -> None:
        self.df=df

    def dbscan(self)->pd.DataFrame:
        """
        The goal of this function is to perform
        the DBSCAN clustering algorithm on the 
        preprocessed data
        
        Arguments:
            -None
            
        Returns:
            -self.df: pd.DataFrame: The DataFrame
            after clustering was performed on its
            preprocessed data
        """

        self.df["title_dbscan"]=hero.do_dbscan(self.df["title"])
        self.df['label'] = pd.Categorical(self.df['title_dbscan'],
                  categories=self.df['title_dbscan'].sort_values().unique())

        return self.df
