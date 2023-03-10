# The goal of this python file is to create an embedding model
# taking into account the different languages
import pandas as pd
import re
import sys
import os
import nltk
import simplemma
import torch
import texthero as hero
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

sys.path.insert(
    0, os.path.join(os.getcwd(), "Learning_equality_curriculum_recommendation/loading")
)

sys.path.insert(
    0, os.path.join(os.getcwd(), "Learning_equality_curriculum_recommendation/logs")
)

from loading import *
from logs import *

main()

tqdm.pandas()
dict_language = load_file("dict_language.json")


class Preprocessor:
    """
    The goal of this class is to extract the right textual
    information from the DataFrame and having them ready for
    the embedding
    """

    def __init__(self, df: pd.DataFrame) -> None:
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
        logging.info("The user has launched the preprocessing")

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
        logging.info("Cleaning missing values...")
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
        logging.info("Removing punctuations...")
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
        logging.info("Tokenization in progress...")
        self.df.title = self.df.title.progress_apply(lambda x: word_tokenize(x))
        self.df.description = self.df.description.progress_apply(
            lambda x: word_tokenize(x)
        )
        self.df.text = self.df.text.progress_apply(lambda x: word_tokenize(x))

        return self.df

    def removing_stopwords_per_sentence(self, sentence: list, language: str) -> str:
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
            stop = set(stopwords.words(dict_language[language]))
            sentence = [x for x in sentence if x not in stop]
            return sentence
        else:
            return sentence

    def remove_stopwords_cleaning(self) -> pd.DataFrame():
        """
        The goal of this function is to apply, per sentence,
        the stopword cleaning in order to have a DataFrame
        with exploitable textual data

        Arguments:
            -None

        Returns:
            -self.df: pd.DataFrame: The cleaned DataFrame once
            the stopwords were removed
        """
        logging.info("Removing stopwords...")

        for language in tqdm(dict_language.keys()):
            self.df.loc[self.df.language == language, "title"] = self.df.loc[
                self.df.language == language, "title"
            ].apply(lambda x: self.removing_stopwords_per_sentence(x, language))
            self.df.loc[self.df.language == language, "description"] = self.df.loc[
                self.df.language == language, "description"
            ].apply(lambda x: self.removing_stopwords_per_sentence(x, language))
            self.df.loc[self.df.language == language, "text"] = self.df.loc[
                self.df.language == language, "text"
            ].apply(lambda x: self.removing_stopwords_per_sentence(x, language))

        return self.df

    def lemmatization(self) -> pd.DataFrame():
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

        logging.info("Lemmatization in progress...")
        language_list = [
            "bg",
            "cs",
            "da",
            "en",
            "es",
            "fr",
            "it",
            "pl",
            "ru",
            "sw",
            "tr",
        ]

        for language in tqdm(language_list):
            self.df.loc[self.df.language == language, "title"] = self.df.loc[
                self.df.language == language, "title"
            ].apply(
                lambda x: [simplemma.lemmatize(token, lang=language) for token in x]
            )
            self.df.loc[self.df.language == language, "description"] = self.df.loc[
                self.df.language == language, "description"
            ].apply(
                lambda x: [simplemma.lemmatize(token, lang=language) for token in x]
            )
            self.df.loc[self.df.language == language, "text"] = self.df.loc[
                self.df.language == language, "text"
            ].apply(
                lambda x: [simplemma.lemmatize(token, lang=language) for token in x]
            )

        return self.df

    def stemming(self) -> pd.DataFrame():

        logging.info("Stemming in progress...")
        for language in tqdm(dict_language.keys()):
            try:
                self.df.loc[self.df.language == language, "title"] = self.df.loc[
                    self.df.language == language, "title"
                ].apply(
                    lambda x: [
                        nltk.SnowballStemmer(dict_language[language]).stem(token)
                        for token in x
                    ]
                )
                self.df.loc[self.df.language == language, "description"] = self.df.loc[
                    self.df.language == language, "description"
                ].apply(
                    lambda x: [
                        nltk.SnowballStemmer(dict_language[language]).stem(token)
                        for token in x
                    ]
                )
                self.df.loc[self.df.language == language, "text"] = self.df.loc[
                    self.df.language == language, "text"
                ].apply(
                    lambda x: [
                        nltk.SnowballStemmer(dict_language[language]).stem(token)
                        for token in x
                    ]
                )
            except:
                pass
        return self.df

    def overall_preprocessing(self) -> pd.DataFrame:
        """
        The goal of this function is to apply the all
        preprocessing steps of the pipeline to the
        entered DataFrame

        Arguments:
            -None

        Returns:
            -self.df: pd.DataFrame: The DataFrame once
            it has been processed
        """

        self.df = self.cleaning_missing_values()
        self.df = self.removing_punctuation()
        self.df = self.tokenization()
        self.df = self.remove_stopwords_cleaning()
        self.df = self.lemmatization()
        self.df = self.stemming()

        logging.warning("The preprocessing is over")
        return self.df


class Embedding(Preprocessor):
    """
    The goal of this class is to inherit from a preprocessed
    DataFrame with textual information ready to be embedded
    with different methods (tf_idf, word_2_vec)
    """

    def __init__(self, df) -> None:
        self.df = df
        logging.info("The embedding has been launched by the user")

    def tf_idf(self) -> pd.DataFrame():
        """
        The goal of this function is to
        embed words with the tf_idf technique
        and to return vectors from words ready
        to be used

        Arguments:
            -None

        Returns:
            -self.df: pd.DataFrame(): The DataFrame
            with embedded word vectors ready to be used
        """

        logging.info("The TF-IDF embedding has been chosen and performing")
        self.df.title = self.df.title.apply(lambda x: " ".join(x))
        self.df.description = self.df.description.apply(lambda x: " ".join(x))
        self.df.title = hero.do_tfidf(self.df.title)
        self.df.description = hero.do_tfidf(self.df.description)

        return self.df

    def word_2_vec(self):
        """
        The goal of this function is to
        embed words with the word_2_vec technique
        and to return vectors from words ready
        to be used

        Arguments:
            -None

        Returns:
            -self.df: pd.DataFrame(): The DataFrame
            with embedded word vectors ready to be used
        """
        logging.info("The word 2 vec embedding has been chosen and performing")
        pass

    def bert_embedder(self):
        logging.info("The BERT embedding has been chosen and performing")
        model = BertModel.from_pretrained("bert-base-uncased")
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        def bert_tokenizer(text):
            tokens = tokenizer.tokenize(text)
            tokens = ["[CLS]"] + tokens + ["[SEP]"]
            inputs = tokenizer.convert_tokens_to_ids(tokens)
            inputs = torch.tensor([inputs])
            outputs = model(inputs)
            hidden_states = outputs[2]
            word_embeddings = hidden_states[-1][0]
            return word_embeddings

        self.df.title = self.df.title.progress_apply(lambda x: bert_tokenizer(x))
        self.df.description = self.df.description.progress_apply(
            lambda x: bert_tokenizer(x)
        )

        return self.df


if __name__ == "__main__":
    content = pd.read_csv("data/content.csv")
    embedder = Embedding(content)
    embedder.bert_embedder()
