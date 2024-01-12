import joblib
import lightgbm as lgb
import optuna
import pandas as pd

from config.config import Config
from lgbm import LightGBM


class ClickLGBM(LightGBM):

    def __init__(self):
        super().__init__(is_regression=False,
                         target='click', objective='binary', metric='binary', cv_metric='valid binary_logloss-mean',
                         n_estimators=5000, learning_rate=0.00873,
                         num_leaves=17, feature_fraction=0.646, lambda_l1=0.79814, lambda_l2=6.42049,
                         bagging_fraction=0.847, bagging_freq=2, min_data_in_leaf=20, early_stopping_round=100,
                         cv=5)

        self.features = ['fc_imp_chk',
                         'fc_time_chk',
                         'utmtr',
                         'mm_dma',
                         'osName',
                         'model',
                         'hardware',
                         'domain',
                         'site_category',
                         'year',
                         'month',
                         'hour',
                         'week_day'
                         ]

        self.cat_features = ['mm_dma', 'osName', 'model', 'hardware', 'domain', 'week_day', 'site_category']
        self.columns = self.features + [self.target]

        self.train_path = Config().data_paths['train']
        self.validation_path = Config().data_paths['validation']

        self.model_path = Config().model_paths['lgbm']

    def optuna_optimization(self, n_trials: int) -> pd.DataFrame:
        """"""

        study = optuna.create_study(direction="minimize")

        train = pd.read_feather(self.train_path, columns=self.columns)
        x = train.loc[:, self.features]
        y = train[self.target]

        train_set = lgb.Dataset(x, y, categorical_feature=self.cat_features)

        fraction_step = round((1 / len(self.features)) - 0.001, 4)

        def objective(trial):
            params = {
                'objective': self.objective,
                'n_estimators': self.n_estimators,
                'verbose': -1,

                'early_stopping_round': self.early_stopping_round,
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.02, step=0.000001),

                'feature_fraction': trial.suggest_float('feature_fraction', 0.45, 0.75, step=fraction_step),
                'num_leaves': trial.suggest_int('num_leaves', 10, 30),
                'lambda_l1': trial.suggest_float('lambda_l1', 0.5, 4, step=0.00001),
                'lambda_l2': trial.suggest_float('lambda_l2', 1, 10, step=0.00001),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.7, 0.95, step=fraction_step),
                'bagging_freq': trial.suggest_int('bagging_freq', 2, 200, step=2),
                # 'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 50),

            }

            return self.regression_cv_score(params, train_set)

        study.optimize(objective, n_trials=n_trials, gc_after_trial=True, show_progress_bar=True)

        experiments = study.trials_dataframe()

        experiments = (experiments
                       .drop(columns=['datetime_start', 'datetime_complete', 'state'])
                       .sort_values(['value'], ascending=True))

        experiments.columns = [col.replace('params_', '') for col in experiments.columns]

        return experiments

    def predict(self, new_data=None) -> pd.DataFrame:
        """Predicts the outcomes of the football matches."""

        if new_data is None:
            new_data = pd.read_feather(self.test_path)

        model = joblib.load(self.model_path)

        new_data['click_probability'] = model.predict_proba(new_data.loc[:, self.features])[:, 1]

        return new_data
