#The goal of this python file is to perform unsupervised analysis 
#on data that has just been preprocessed

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
    def __init__(self,df:pd.DataFrame) -> None:
        self.df=df

    def pca(self)->pd.DataFrame:
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
        
        self.df['clean_title_pca'] = hero.do_pca(self.df['title'])

        return self.df

    def visualize_pca(self):
        hero.scatterplot(self.df, col='clean_title_pca', color='language', title="PCA visualisation of language distribution")