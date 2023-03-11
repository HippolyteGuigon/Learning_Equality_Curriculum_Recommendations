import pandas as pd 
import numpy as np
import os
import transformers
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoConfig
from torch.utils.data import DataLoader, Dataset
from transformers import get_cosine_schedule_with_warmup, DataCollatorWithPadding
from sklearn.neighbors import NearestNeighbors
from Learning_equality_curriculum_recommendation.configs.confs import (
    load_conf,
    clean_params,
    Loader,
)

os.environ["TOKENIZERS_PARALLELISM"]="False"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tqdm.pandas()

main_params = load_conf("configs/main.yml", include=True)
main_params = clean_params(main_params)
unsupervised_model = main_params["unsupervised_model"]
batch_size = main_params["batch_size"]
num_workers=main_params["num_workers"]
top_n = main_params["top_n"]

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

class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings
    
class uns_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = AutoConfig.from_pretrained(unsupervised_model)
        self.model = AutoModel.from_pretrained(unsupervised_model, config = self.config)
        self.pool = MeanPooling()
    def feature(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_state = outputs.last_hidden_state
        feature = self.pool(last_hidden_state, inputs['attention_mask'])
        return feature
    def forward(self, inputs):
        feature = self.feature(inputs)
        return feature
    
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
        
    def get_embeddings(self, loader, model, device):
        model.eval()
        preds = []
        for step, inputs in enumerate(tqdm(loader)):
            for k, v in inputs.items():
                inputs[k] = v.to(device)
            with torch.no_grad():
                y_preds = model(inputs)
            preds.append(y_preds.to('cpu').numpy())
        preds = np.concatenate(preds)
        return preds

    def fit(self):
        model = uns_model()
        model.to(device)
        self.topics_preds = np.array(self.get_embeddings(self.topics_loader, model, device))
        self.content_preds = np.array(self.get_embeddings(self.content_loader, model, device))
        self.neighbors_model = NearestNeighbors(n_neighbors = top_n, metric = 'cosine')
        self.neighbors_model.fit(content_preds)

    def predict(self):
        indices = self.neighbors_model.kneighbors(self.topics_preds, return_distance = False)
        predictions = []
        for k in range(len(indices)):
            pred = indices[k]
            p = ' '.join([self.content.loc[ind, 'id'] for ind in pred.get()])
            predictions.append(p)
        self.topics['predictions'] = predictions
        # Release memory
        return self.topics, self.content 


if __name__=="__main__":
    model=Transformer_model()
    model.get_loader()
    model.fit()