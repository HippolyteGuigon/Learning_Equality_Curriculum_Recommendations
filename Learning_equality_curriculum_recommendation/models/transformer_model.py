import pandas as pd 
import transformers
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
from torch.utils.data import DataLoader, Dataset
from transformers import get_cosine_schedule_with_warmup, DataCollatorWithPadding
from Learning_equality_curriculum_recommendation.configs.confs import (
    load_conf,
    clean_params,
    Loader,
)

main_params = load_conf("configs/main.yml", include=True)
main_params = clean_params(main_params)
unsupervised_model = main_params["unsupervised_model"]
batch_size = main_params["batch_size"]
num_workers=main_params["num_workers"]

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

class uns_dataset(Dataset):
    """
    The goal of this class is to create 
    the unsupervised dataset that will 
    then be used for correlation computation

    Arguments: 
        df: pd.DataFrame: The DataFrame from
        which the unsupervised dataset will be 
        created
    """
    def __init__(self, df: pd.DataFrame)->None:
        self.texts = df['title'].values

    def prepare_uns_input(text):
        inputs = unsupervised_tokenizer.encode_plus(
            text, 
            return_tensors = None, 
            add_special_tokens = True, 
        )
        for k, v in inputs.items():
            inputs[k] = torch.tensor(v, dtype = torch.long)
        return inputs
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, item):
        inputs = unsupervised_tokenizer(self.texts[item])
        return inputs
    

class Transformer_model:
    """
    The goal of this class is to implement 
    a Transformer model in order to find the 
    most correlated content with he topics
    
    Arguments:
        None
    """
    
    def __init__(self):
        self.topics, self.content = read_data()

    def get_loader(self)->torch:
        """
        The goal of this function is to create 
        Loaders set that will pass through the 
        Trasformer model

        Arguments:
            None

        Returns:
            -self.topics_loader: torch: The DataLoader
            of the topics dataset
            -self.content_loader: torch: The DataLoader
            of the content dataset
        """
        topics_dataset = uns_dataset(self.topics)
        content_dataset = uns_dataset(self.content)

        self.topics_loader = DataLoader(
        topics_dataset, 
        batch_size = batch_size, 
        shuffle = False, 
        collate_fn = DataCollatorWithPadding(tokenizer = unsupervised_tokenizer, padding = 'longest'),
        num_workers = num_workers, 
        pin_memory = True, 
        drop_last = False
    )
        self.content_loader = DataLoader(
            content_dataset, 
            batch_size = batch_size, 
            shuffle = False, 
            collate_fn = DataCollatorWithPadding(tokenizer = unsupervised_tokenizer, padding = 'longest'),
            num_workers = num_workers, 
            pin_memory = True, 
            drop_last = False
            )
        
        return self.topics_loader, self.content_loader
        

if __name__=="__main__":
    model=Transformer_model()
    model.get_loader()