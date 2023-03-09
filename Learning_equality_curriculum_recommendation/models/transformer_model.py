import pandas as pd 
import transformers
from transformers import AutoTokenizer, AutoModel, AutoConfig
from Learning_equality_curriculum_recommendation.configs.confs import (
    load_conf,
    clean_params,
    Loader,
)

main_params = load_conf("configs/main.yml", include=True)
main_params = clean_params(main_params)
unsupervised_model = main_params["unsupervised_model"]

unsupervised_tokenizer = AutoTokenizer.from_pretrained(unsupervised_model)

def read_data()->pd.DataFrame:
    """
    The goal of this function is to load the 
    different DataFrames that will be used for 
    the transformer model
    
    Arguments:
        None
        
    Returns:
        -topics: pd.DataFrame: The topics DataFrame
        with only the topics_id that are inside the 
        submission file with titles
        -content: pd.DataFrame: The content DataFrame
        with titles
    """

    topics = pd.read_csv('data/topics.csv')
    content = pd.read_csv('data/content.csv')
    sample_submission = pd.read_csv('data/sample_submission.csv')
    topics = topics.merge(sample_submission, how = 'inner', left_on = 'id', right_on = 'topic_id')
    topics['title'].fillna("", inplace = True)
    content['title'].fillna("", inplace = True)
    topics['length'] = topics['title'].apply(lambda x: len(x))
    content['length'] = content['title'].apply(lambda x: len(x))
    topics.sort_values('length', inplace = True)
    content.sort_values('length', inplace = True)
    topics.drop(['description', 'channel', 'category', 'level', 'language', 'parent', 'has_content', 'length', 'topic_id', 'content_ids'], axis = 1, inplace = True)
    content.drop(['description', 'kind', 'language', 'text', 'copyright_holder', 'license', 'length'], axis = 1, inplace = True)
    topics.reset_index(drop = True, inplace = True)
    content.reset_index(drop = True, inplace = True)
    return topics, content