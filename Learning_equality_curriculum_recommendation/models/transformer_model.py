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
supervised_model = main_params["supervised_model"]
supervised_model_tuned = main_params["supervised_model_tuned"]
gradient_checkpointing = main_params["gradient_checkpointing"]
batch_size = main_params["batch_size"]
num_workers=main_params["num_workers"]
top_n = main_params["top_n"]

unsupervised_tokenizer = AutoTokenizer.from_pretrained(unsupervised_model)
supervised_tokenizer = AutoTokenizer.from_pretrained(supervised_model)

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
    

class custom_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = AutoConfig.from_pretrained(supervised_model_tuned + '/config', output_hidden_states = True)
        self.config.hidden_dropout = 0.0
        self.config.hidden_dropout_prob = 0.0
        self.config.attention_dropout = 0.0
        self.config.attention_probs_dropout_prob = 0.0
        self.model = AutoModel.from_pretrained(supervised_model_tuned + '/model', config = self.config)
        #self.pool = MeanPooling()
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.pool = MeanPooling()
        
        #self.fc = nn.Linear(self.config.hidden_size, 1)
        self._init_weights(self.fc)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    def feature(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_state = outputs.last_hidden_state
        feature = self.pool(last_hidden_state, inputs['attention_mask'])
        
        #last_hidden_state = outputs.last_hidden_state
        #feature = self.pool(last_hidden_state, inputs['attention_mask'])
        return feature
    
    def forward(self, inputs):
        feature = self.feature(inputs)
        output = self.fc(feature)
        return output

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

def prepare_sup_input(text):
    inputs = supervised_tokenizer.encode_plus(
        text, 
        return_tensors = None, 
        add_special_tokens = True, 
    )
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype = torch.long)
    return inputs

class sup_dataset(Dataset):
    def __init__(self, df):
        self.texts = df['text'].values
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, item):
        inputs = prepare_sup_input(self.texts[item])
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
        """
        The goal of this function is to 
        
        Arguments: 
            -loader: torch: The batches of data 
            that will be passed through the model 
            -model: torch: The model that will embedd
            the different string sentences passed through
            the model
            -device: torch: The device on which the model 
            will be run
            
        Returns: 
            -preds: np.array: The predictions
            made for the closest sentences given 
            a certain input
        """
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
        """
        The goal of this function is, once the embeddings
        are produced, to fit a Nearest Neighbours model on 
        it that will be able to detect the closest sentences
        
        Arguments: 
            None 
            
        Returns:
            None"""
        model = uns_model()
        model.to(device)
        self.topics_preds = np.array(self.get_embeddings(self.topics_loader, model, device))
        self.content_preds = np.array(self.get_embeddings(self.content_loader, model, device))
        self.neighbors_model = NearestNeighbors(n_neighbors = top_n, metric = 'cosine')
        self.neighbors_model.fit(self.content_preds)

    def predict(self):
        """
        The goal of this function is, once the 
        model is fitted, to get the predictions, 
        the content that are closest for a given 
        input 
        
        Arguments: 
            None
            
        Returns:
            -self.topics: pd.DataFrame: The topics 
            DataFrame with the closest topics
            -self.content: """
        indices = self.neighbors_model.kneighbors(self.topics_preds, return_distance = False)
        predictions = []
        for k in range(len(indices)):
            pred = indices[k]
            p = ' '.join([self.content.loc[ind, 'id'] for ind in pred])
            predictions.append(p)
        self.topics['predictions'] = predictions
        # Release memory
        return self.topics, self.content 

    def build_inference_set(self, topics, content):
        # Create lists for training
        topics_ids = []
        content_ids = []
        topics_languages = []
        content_languages = []
        title1 = []
        title2 = []
        has_contents = []
        # Iterate over each topic
        full_topics=pd.read_csv("data/topics.csv")
        full_content=pd.read_csv("data/content.csv")
        topics=topics.merge(full_topics[["id", "language", "has_content"]], on="id", how="left")
        content=content.merge(full_content[["id", "language"]], on="id", how="left")
        content.set_index("id",inplace=True)
        for k in tqdm(range(len(topics))):
            row = topics.iloc[k]
            topics_id = row['id']
            topics_language = row['language']
            topics_title = row['title']
            predictions = row['predictions'].split(' ')
            has_content = row['has_content']
            for pred in predictions:
                content_title = content.loc[pred, 'title']
                content_language = content.loc[pred, 'language']
                topics_ids.append(topics_id)
                content_ids.append(pred)
                title1.append(topics_title)
                title2.append(content_title)
                topics_languages.append(topics_language)
                content_languages.append(content_language)
                has_contents.append(has_content)
                
        # Build training dataset
        test = pd.DataFrame(
            {'topics_ids': topics_ids, 
            'content_ids': content_ids, 
            'title1': title1, 
            'title2': title2,
            'topic_language': topics_languages, 
            'content_language': content_languages,
            'has_contents': has_contents,
            }
        )
        
        return test
    
    def preprocess_test(self, tmp_test):
        tmp_test['title1'].fillna("Title does not exist", inplace = True)
        tmp_test['title2'].fillna("Title does not exist", inplace = True)
        # Create feature column
        tmp_test['text'] = tmp_test['title1'] + '[SEP]' + tmp_test['title2']
        # Drop titles
        tmp_test.drop(['title1', 'title2'], axis = 1, inplace = True)
        # Sort so inference is faster
        tmp_test['length'] = tmp_test['text'].apply(lambda x: len(x))
        tmp_test.sort_values('length', inplace = True)
        tmp_test.drop(['length'], axis = 1, inplace = True)
        tmp_test.reset_index(drop = True, inplace = True)
        return tmp_test

    def inference_fn(self, test_loader, model, device):
        preds = []
        model.eval()
        model.to(device)
        tk0 = tqdm(test_loader, total = len(test_loader))
        for inputs in tk0:
            for k, v in inputs.items():
                inputs[k] = v.to(device)
            with torch.no_grad():
                y_preds = model(inputs)
            preds.append(y_preds.sigmoid().squeeze().to('cpu').numpy().reshape(-1))
        predictions = np.concatenate(preds)
        return predictions

    def inference(self, test, _idx):
        # Create dataset and loader
        test_dataset = sup_dataset(test)
        test_loader = DataLoader(
            test_dataset, 
            batch_size = batch_size, 
            shuffle = False, 
            collate_fn = DataCollatorWithPadding(tokenizer = supervised_tokenizer, padding = 'longest'),
            num_workers = 8,
            pin_memory = True,
            drop_last = False
        )
        # Get model
        model = custom_model()
        
        # Load weights
        state = torch.load(supervised_model_tuned, map_location = torch.device('cpu'))
        model.load_state_dict(state['model'])
        prediction = model.inference_fn(test_loader, model, device)
                
        # Use threshold
        test['probs'] = prediction
        test['predictions'] = test['probs'].apply(lambda x: int(x > 0.1))
        #test['predictions'] = test['probs'].apply(lambda x: int(x > 0.001)) 
        test = test.merge(test.groupby("topics_ids", as_index=False)["probs"].max(), on="topics_ids", suffixes=["", "_max"])
        test = test[test['has_contents'] == True]
        #display(test)
        
        test1 = test[(test['predictions'] == 1) & (test['topic_language'] == test['content_language'])]
        test1 = test1.groupby(['topics_ids'])['content_ids'].unique().reset_index()
        test1['content_ids'] = test1['content_ids'].apply(lambda x: ' '.join(x))
        test1.columns = ['topic_id', 'content_ids']
        #display(test1.head())
        
        test0 = pd.Series(test['topics_ids'].unique())
        test0 = test0[~test0.isin(test1['topic_id'])]
        test0 = pd.DataFrame({'topic_id': test0.values, 'content_ids': ""})
        #display(test0.head())
        test_r = pd.concat([test1, test0], axis = 0, ignore_index = True)
        test_r.to_csv(f'submission_{_idx+1}.csv', index = False)
        
        return test_r

if __name__=="__main__":
    model=Transformer_model()
    tmp_topics, tmp_content = pd.read_csv("topics_pred.csv"), pd.read_csv("content_pred.csv")
    tmp_content.set_index('id', inplace = True)
    tmp_test = model.build_inference_set(tmp_topics, tmp_content)
    tmp_test = model.preprocess_test(tmp_test)
    print("AT THIS STAGE 1")
    model.inference(tmp_test, 1)
    print("AT THIS STAGE 2")    
    df_test = pd.read_csv('submission_1.csv')
    df_test.fillna("", inplace = True)
    df_test['content_ids'] = df_test['content_ids'].apply(lambda c: c.split(' '))
    df_test = df_test.explode('content_ids').groupby(['topic_id'])['content_ids'].unique().reset_index()
    df_test['content_ids'] = df_test['content_ids'].apply(lambda c: ' '.join(c))
    df_test.to_csv('submission.csv', index = False)
    df_test.head()