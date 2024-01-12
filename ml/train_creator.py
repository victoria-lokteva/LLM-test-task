import pandas as pd
from ml.preprocessing import DataPreprocessor
from sklearn.model_selection import train_test_split
from config.config import Config


class TrainCreator(object):

    def __init__(self, test_size=0.2, random_state=7):
        self.test_size = test_size
        self.random_state = random_state

    def transform(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split the data into train and validation sets."""
        data = DataPreprocessor().preprocessing()

        # remove not features columns
        data = data.drop(columns=['tag', 'site_id'])

        # categorical features
        cat_features = ['osName', 'model', 'hardware', 'domain', 'week_day', 'site_category']
        data[cat_features] = data[cat_features].astype('category')

        train, validation = train_test_split(data, test_size=self.test_size, random_state=self.random_state,
                                             shuffle=True, stratify=data['click'])

        train.reset_index().to_feather(Config().data_paths['train'])
        validation.reset_index().to_feather(Config().data_paths['validation'])

        return train, validation
