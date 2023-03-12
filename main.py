from Learning_equality_curriculum_recommendation.configs.confs import (
    load_conf,
    clean_params,
    Loader,
)

main_params = load_conf("configs/main.yml", include=True)
main_params = clean_params(main_params)

chosen_model = main_params["chosen_model"]

def launch_model():
    pass

def fill_submission_file():
    pass