# Learning_Equality_Curriculum_Recommendations

The goal of this repository is to participate to the Learning Equality Curriculum Recommendation competition in Kaggle (https://www.kaggle.com/competitions/learning-equality-curriculum-recommendations). The goal of this competition is to train an algorithm to detect similarities between the contents of school curricula in different countries in order to make them more quickly and easily accessible to teachers and students. 

## Build Status

For the moment, two models are ready to be used in the main pipeline: Sentence Bert model and a Transformer encoder used with KNN. 

The objective for the rest of the project is to fine tune the hyperparameters while making the overall utilisation very intuitive and user friendly.

Throughout its construction, if you see any improvements that could be made in the code, do not hesitate to reach out at 
Hippolyte.guigon@hec.edu

## Code style 

The all project was coded under PEP-8 (https://peps.python.org/pep-0008/) and flake8 (https://pypi.org/project/flake8/) compliancy. Such compliance is verified during commits with pre-commits file ```.pre-commit-config.yaml```

## Installation

* This project uses a specific conda environment, to get it, run the following command: ```conda env create -f learning_equality_curriculum_recommendation.yml```

* To install all necessary libraries, run the following code: ```pip install -r requirements.txt```

* This project has its own package that is used. To get it, run the following command: ```python install setup.py```

## Screenshot 

![alt text](https://github.com/HippolyteGuigon/Learning_Equality_Curriculum_Recommendations/blob/features_building_embedding/ressources/Educational_recommendation_system.png)

Recommendation of an educational ressource for a given research

## How to use ? 

For the moment, the main.py file is still under construction. 