import optuna
import pandas as pd

from catboost_model import CatBoost, CatBoostClassifier, Pool
from config.config import Config


class ClickCatBoost(CatBoost):

    def __init__(self, target='click'):
        super().__init__(is_regression=False,
                         target=target,
                         fold_count=5,
                         iterations=5000, learning_rate=0.05090175934770157, colsample_bylevel=0.9092111469100367,
                         depth=7, l2_leaf_reg=3.4802473839385946, bagging_temperature=1,
                         random_strength=4.356977387977597, od_wait=100, min_data_in_leaf=18)

        self.loss_function = 'Logloss'
        self.cv_metric = 'test-Logloss-mean'
        self.bootstrap_type = 'MVS'
        self.boosting_type = 'Plain'

        self.train_path = Config().data_paths['train']
        self.validation_path = Config().data_paths['validation']

        self.model_path = Config().model_paths['cb']

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

    def optuna_optimization(self, n_trials: int) -> pd.DataFrame:
        """"""

        study = optuna.create_study(direction="minimize")

        train = pd.read_feather(self.train_path)

        cv_dataset = Pool(data=train.loc[:, self.features],
                          label=train[self.target],
                          cat_features=self.cat_features)

        def objective(trial):
            """boosting_type:
            Ordered — Usually provides better quality on small datasets (< 50k), but it may be slower than the Plain scheme.
            Plain — The classic gradient boosting scheme."""

            params = {
                'loss_function': self.loss_function,
                'od_type': self.od_type,
                'iterations': self.iterations,
                "boosting_type": self.boosting_type,
                'verbose': False,
                "allow_writing_files": False,

                'od_wait': self.od_wait,
                'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.08),

                "l2_leaf_reg": trial.suggest_loguniform("l2_leaf_reg", 0.01, 5),
                "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.6, 1),
                "depth": trial.suggest_int("depth", 6, 12),

                "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]),
                'min_data_in_leaf': trial.suggest_int("min_data_in_leaf", 5, 50),
                "random_strength": trial.suggest_float("random_strength", 0.1, 5),

            }

            if params["bootstrap_type"] == "Bayesian":
                params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
            elif params["bootstrap_type"] == "Bernoulli":
                params["subsample"] = trial.suggest_float("subsample", 0.1, 1)

            return self.optuna_cv_score(params, cv_dataset)

        study.optimize(objective, n_trials=n_trials, n_jobs=1, gc_after_trial=True, show_progress_bar=True)

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

        model = CatBoostClassifier()

        model.load_model(self.model_path)

        new_data['click_probability'] = model.predict_proba(new_data.loc[:, self.features])[:, 1]

        return new_data
