import pandas as pd


class DataPreprocessor(object):

    def __init__(self):
        self.train_path = "data/interview.X.csv"
        self.test_path = "data/interview.y.csv"

    def preprocessing(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """"""
        train = pd.read_csv(self.train_path)
        test = pd.read_csv(self.test_path)

        train = train.drop_duplicates()
        test = test.drop_duplicates()

        # fill nans for categorical features
        for col in ['osName', 'model', 'hardware']:
            train.loc[train[col].isna(), col] = 'unknown'

        train = self._website_name_preprocessing(train)

        # значения dma, которые встречаются менее 5 раз, объединим в одну категорию -1 ("другое")
        rare_dma = [dma for dma, count in train['mm_dma'].value_counts().items() if count < 5]
        train.loc[train['mm_dma'].isin(rare_dma), 'mm_dma'] = -1

        # Аналогично поступим с site_id, model
        rare_site = [site for site, count in train['site_id'].value_counts().items() if count < 5]
        train.loc[train['site_id'].isin(rare_site), 'site_id'] = "other"

        rare_model = [model for model, count in train['model'].value_counts().items() if count < 5]
        train.loc[train['model'].isin(rare_model), 'model'] = "other"

        data = train.merge(test, how='inner', on='uid')
        data = self._date_preprocessing(data)
        data = self._targets(data)

        data.to_csv("/Users/victorialokteva/Downloads/interview.csv")

        return train, test, data

    @staticmethod
    def _date_preprocessing(data: pd.DataFrame) -> pd.DataFrame:
        """"""
        # convert strings into dates
        data['reg_time'] = pd.to_datetime(data['reg_time'], format='%Y-%m-%d %H:%M:%S')

        data['month'] = data['reg_time'].dt.month
        data['hour'] = data['reg_time'].dt.hour
        data['day'] = pd.to_datetime(data['reg_time']).dt.day_name()
        return data

    @staticmethod
    def _targets(data: pd.DataFrame) -> pd.DataFrame:
        """"""
        # ???
        # событие fclick - первый клик (используется для расчета CTR).
        data['click'] = (data['tag'] == 'fclick')
        return data

    @staticmethod
    def _website_name_preprocessing(train: pd.DataFrame) -> pd.DataFrame:
        """Extract information from website names.
        Add categories such as sports and health based on keywords."""

        # remove www, www1
        train['site_id'] = train['site_id'].map(lambda x: x.replace("www1.", "").replace("www.", ""))

        # add a feature with domain-related information.
        train['domain'] = train['site_id'].map(lambda x: x.split('.')[-1])

        #
        train['is_sport_site'] = (train['site_id'].str.contains('sport')
                                  | train['site_id'].str.contains('baseball')
                                  | train['site_id'].str.contains('football')
                                  | train['site_id'].str.contains('soccer')
                                  | train['site_id'].str.contains('basketball')
                                  | train['site_id'].str.contains('hockey')
                                  | train['site_id'].str.contains('moneyball')
                                  | train['site_id'].str.contains('rotowire')
                                  | train['site_id'].str.contains('mmafighting')
                                  | train['site_id'].str.contains('mmamania')
                                  | train['site_id'].str.contains('nbaanalysis')
                                  | train['site_id'].str.contains('nba-trade-rumors')
                                  | train['site_id'].str.contains('nfltraderumors')
                                  | train['site_id'].str.contains('nfldraftdiamonds')
                                  | train['site_id'].str.contains('detroitbadboys.com')  # ?
                                  | train['site_id'].str.contains('volleyball'))

        train['is_travel_site'] = (train['site_id'].str.contains('travel')
                                   | train['site_id'].str.contains('tourist'))

        train['is_weather_site'] = (train['site_id'].str.contains('weather'))

        train['is_game_site'] = (train['site_id'].str.contains('game'))

        train['is_music_site'] = (train['site_id'].str.contains('music')
                                  | train['site_id'].str.contains('hip-hop')
                                  | train['site_id'].str.contains('song')  # ?
                                  | train['site_id'].str.contains('guitar'))

        train['is_cook_site'] = (train['site_id'].str.contains('cook')
                                 | train['site_id'].str.contains('diet')
                                 | train['site_id'].str.contains('fastfood')
                                 | train['site_id'].str.contains('cupcake')  # javacupcake.com
                                 | train['site_id'].str.contains('grandbaby-cakes')
                                 | train['site_id'].str.contains('browneyedbaker')  # baker?
                                 | train['site_id'].str.contains('baking')
                                 | train['site_id'].str.contains('lovebakesgoodcakes')
                                 | train['site_id'].str.contains('bakery'))

        train['is_health_site'] = (train['site_id'].str.contains('health')
                                   | train['site_id'].str.contains('medical'))

        train['is_zodiac_site'] = (train['site_id'].str.contains('astrology')
                                   | train['site_id'].str.contains('zodiac'))

        return train
