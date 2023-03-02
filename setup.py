from setuptools import setup, find_packages

setup(
    name="Hispatologic_cancer_detection",
    version="0.1.0",
    packages=find_packages(
        include=["Learning_equality_curriculum_recommendation", "Learning_equality_curriculum_recommendation.*"]
    ),
    description="Python programm for the Kaggle competition\
        Learning_equality_curriculum_recommendation",
    author="Hippolyte Guigon",
    author_email="Hippolyte.guigon@hec.edu",
    url="https://github.com/HippolyteGuigon/Learning_Equality_Curriculum_Recommendations",
)