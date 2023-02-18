# The goal of this python file is to create an embedding model
# taking into account the different languages
import pandas as pd
import re
import sys
import os
import nltk
import simplemma
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from tqdm import tqdm

sys.path.insert(
    0, os.path.join(os.getcwd(), "Learning_equality_curriculum_recommendation/loading")
)
from loading import *

tqdm.pandas()
dict_language = load_file("dict_language.json")


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

    def removing_punctuation(self) -> pd.DataFrame():
        """
        The goal of this function is to remove all punctuations
        from the textual data within the DataFrame

        Arguments:
            -None

        Returns:
            -self.df: The DataFrame newly cleaned from punctuation
        """
        self.df.title = self.df.title.progress_apply(
            lambda x: re.sub(r"[^\w\s]", "", x)
        )
        self.df.description = self.df.description.progress_apply(
            lambda x: re.sub(r"[^\w\s]", "", x)
        )
        self.df.text = self.df.text.progress_apply(lambda x: re.sub(r"[^\w\s]", "", x))

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
        self.df.text = self.df.text.progress_apply(lambda x: word_tokenize(x))

        return self.df

    def removing_stopwords_per_sentence(self, sentence: list, language:str) -> str:
        """
        The goal of this function is removing stopword
        (words useless for the context) in the DataFrame
        textual data

        Arguments:
            -sentence: list: The list of words within the 
            sentence 
            -language: str: The language in which the sentence
            is written

        Returns:
            -sentence: string: The sentence after it was cleaned
            from the stopwords 
        """

        if language in dict_language.keys():
            stop=set(stopwords.words(dict_language[language]))
            sentence = [x for x in sentence if x not in stop]
            return sentence
        else:
            return sentence

    def remove_stopwords_cleaning(self)->pd.DataFrame():
        """
        The goal of this function is to apply, per sentence, 
        the stopword cleaning in order to have a DataFrame 
        with exploitable textual data
        
        Arguments: 
            -None
            
        Returns:
            -self.df: pd.DataFrame: The cleaned DataFrame once 
            the stopwords were removed"""
        for language in tqdm(dict_language.keys()):
            self.df.loc[self.df.language==language,"title"]=self.df.loc[self.df.language==language,"title"].apply(lambda x: self.removing_stopwords_per_sentence(x,language))
            self.df.loc[self.df.language==language,"description"]=self.df.loc[self.df.language==language,"description"].apply(lambda x: self.removing_stopwords_per_sentence(x,language))
            self.df.loc[self.df.language==language,"text"]=self.df.loc[self.df.language==language,"text"].apply(lambda x: self.removing_stopwords_per_sentence(x,language))

        return self.df

    def lemmatization(self)->pd.DataFrame():
        """
        The goal of this function is to apply, per sentence, 
        the lemmatization cleaning in order to have a DataFrame 
        with exploitable textual data
        
        Arguments: 
            -None
            
        Returns:
            -self.df: pd.DataFrame: The cleaned DataFrame once 
            lemmatization was applied
        """
        language_list=["bg","cs","da","en", "es", "fr", "it", 
        "pl", "ru", "sw", "tr"]

        for language in tqdm(language_list):
            self.df.loc[self.df.language==language,"title"]=self.df.loc[self.df.language==language,"title"].apply(lambda x: [simplemma.lemmatize(token, lang=language) for token in x])
            self.df.loc[self.df.language==language,"description"]=self.df.loc[self.df.language==language,"description"].apply(lambda x: [simplemma.lemmatize(token, lang=language) for token in x])
            self.df.loc[self.df.language==language,"text"]=self.df.loc[self.df.language==language,"text"].apply(lambda x: [simplemma.lemmatize(token, lang=language) for token in x])

        return self.df

    def stemming(self)->pd.DataFrame():
        for language in tqdm(dict_language.keys()):
            try:
                self.df.loc[self.df.language==language,"title"]=self.df.loc[self.df.language==language,"title"].apply(lambda x: [nltk.SnowballStemmer(dict_language[language]).stem(token) for token in x])
                self.df.loc[self.df.language==language,"description"]=self.df.loc[self.df.language==language,"description"].apply(lambda x: [nltk.SnowballStemmer(dict_language[language]).stem(token) for token in x])
                self.df.loc[self.df.language==language,"text"]=self.df.loc[self.df.language==language,"text"].apply(lambda x: [nltk.SnowballStemmer(dict_language[language]).stem(token) for token in x])
            except:
                pass
        return self.df

class Embedding(Preprocessor):
    """
    The goal of this class is to inherit from a preprocessed
    DataFrame with textual information ready to be embedded
    """

    def __init__(self, df) -> None:
        self.df = df
