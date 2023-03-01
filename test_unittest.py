import unittest
import pandas as pd
from Learning_equality_curriculum_recommendation.configs.confs import (
    load_conf,
    clean_params,
    Loader,
)
from Learning_equality_curriculum_recommendation.analysis.analysis import (
    single_column_analysis,
)

main_params = load_conf("configs/main.yml", include=True)
main_params = clean_params(main_params)
correlations = pd.read_csv(main_params["correlations_link"])
topics = pd.read_csv(main_params["topics_link"])
content = pd.read_csv(main_params["content_link"])


class Test(unittest.TestCase):
    """
    The goal of this class is to implement unnitest
    and check everything commited makes sense
    """

    def test_single_column_analysis_function(self) -> bool:
        df_analysis_test = single_column_analysis("kind")
        self.assertTrue("content_kind" in df_analysis_test.columns)
        self.assertTrue("topics_kind" in df_analysis_test.columns)


if __name__ == "__main__":
    unittest.main()
