from Learning_equality_curriculum_recommendation.models.sentence_bert import Sentence_Bert_Model
from Learning_equality_curriculum_recommendation.models.transformer_model import Transformer_model
from Learning_equality_curriculum_recommendation.configs.confs import (
    load_conf,
    clean_params,
    Loader,
)
import pandas as pd

sample_submission=pd.read_csv("data/sample_submission.csv")

main_params = load_conf("configs/main.yml", include=True)
main_params = clean_params(main_params)

chosen_model = main_params["chosen_model"]

def launch_model():
    if chosen_model=="transformer_model":
        model=Transformer_model()
    else:
        model=Sentence_Bert_Model(sample_submission)

def fill_submission_file():
    pass

if __name__=="__main__":
    launch_model()
    fill_submission_file()