import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

class TfIdf:
    def __init__(self,
                 max_df: float = 0.95,
                 min_df: int = 2,
                 ngram_range: tuple = (1, 2)
                 ):
        self.vectorizer = TfidfVectorizer(max_df=max_df, min_df=min_df, ngram_range=ngram_range)

    def run(self, df: pd.DataFrame, column_name: str) -> pd.DataFrame:
        return self.vectorizer.fit_transform(df[column_name])

    def count_feature_names_out(self) -> None:
        len(self.vectorizer.get_feature_names_out())

    def to_array(self, tfidf_matrix: pd.DataFrame) -> pd.DataFrame:
        return tfidf_matrix.toarray()