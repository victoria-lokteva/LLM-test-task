import numpy as np
import pandas as pd
from config.config import Config


class DataPreprocessor(object):

    def __init__(self):
        self.x_data = Config().data_paths['x_data']
        self.y_data = Config().data_paths['y_data']

    def preprocessing(self) -> pd.DataFrame:
        """"""
        train = pd.read_csv(self.x_data)
        test = pd.read_csv(self.y_data)

        train = train.drop_duplicates('uid')

        # idea: if the user initiates the first click, subsequent actions are a result of this initial click.
        # therefore, leave uid with "fclick"-tag if there are duplicates
        test['is_first_click'] = (test['tag'] == 'fclick')

        test = (test
                .sort_values(['uid', 'is_first_click'])
                .drop_duplicates('uid', keep='last')
                .drop(columns='is_first_click'))

        data = train.merge(test, how='inner', on='uid')

        # fill nans for categorical features
        for col in ['osName', 'model', 'hardware']:
            data.loc[data[col].isna(), col] = 'unknown'

        # значения dma, которые встречаются менее 5 раз, объединим в одну категорию -1 ("другое")
        rare_dma = [dma for dma, count in data['mm_dma'].value_counts().items() if count < 5]
        data.loc[data['mm_dma'].isin(rare_dma), 'mm_dma'] = -1

        # union Windows users into one category, Symbian and Linux -> other
        data['osName'] = np.where(data['osName'].isin({'Windows 10', 'Windows 7'}), 'Windows', data['osName'])
        data['osName'] = np.where(data['osName'].isin({'Symbian', 'Linux'}), 'other', data['osName'])

        # Аналогично поступим с site_id, model, and osName
        rare_site = [site for site, count in data['site_id'].value_counts().items() if count < 5]
        data.loc[data['site_id'].isin(rare_site), 'site_id'] = "other"

        rare_model = [model for model, count in data['model'].value_counts().items() if count < 5]
        data.loc[data['model'].isin(rare_model), 'model'] = "other"

        data = self._website_name_preprocessing(data)
        data = self._date_preprocessing(data)
        data = self._targets(data)

        data.to_csv("data/dataset.csv")

        return data

    @staticmethod
    def _date_preprocessing(data: pd.DataFrame) -> pd.DataFrame:
        """"""
        # convert strings into dates
        data['reg_time'] = pd.to_datetime(data['reg_time'], format='%Y-%m-%d %H:%M:%S')

        data['year'] = data['reg_time'].dt.year
        data['month'] = data['reg_time'].dt.month

        # similar with utmtr
        data['hour'] = data['reg_time'].dt.hour
        data['week_day'] = pd.to_datetime(data['reg_time']).dt.day_name()

        data = data.drop(columns='reg_time')
        return data

    @staticmethod
    def _targets(data: pd.DataFrame) -> pd.DataFrame:
        """"""
        # событие fclick - первый клик (используется для расчета CTR).
        data['click'] = (data['tag'] == 'fclick')
        return data

    @staticmethod
    def _website_name_preprocessing(data: pd.DataFrame) -> pd.DataFrame:
        """Extract information from website names.
        Add categories such as sports and health based on keywords."""

        # remove www, www1
        data['site_id'] = data['site_id'].map(lambda x: x.replace("www1.", "").replace("www.", ""))

        # add a feature with domain-related information.
        data['domain'] = data['site_id'].map(lambda x: x.split('.')[-1])

        #
        data['is_sport_site'] = (data['site_id'].str.contains('sport')
                                 | data['site_id'].str.contains('baseball')
                                 | data['site_id'].str.contains('football')
                                 | data['site_id'].str.contains('soccer')
                                 | data['site_id'].str.contains('basketball')
                                 | data['site_id'].str.contains('hockey')
                                 | data['site_id'].str.contains('moneyball')
                                 | data['site_id'].str.contains('rotowire')
                                 | data['site_id'].str.contains('mmafighting')
                                 | data['site_id'].str.contains('mmamania')
                                 | data['site_id'].str.contains('nbaanalysis')
                                 | data['site_id'].str.contains('nba-trade-rumors')
                                 | data['site_id'].str.contains('nfltraderumors')
                                 | data['site_id'].str.contains('nfldraftdiamonds')
                                 | data['site_id'].str.contains('detroitbadboys.com')  # ?
                                 | data['site_id'].str.contains('volleyball'))

        data['is_travel_site'] = (data['site_id'].str.contains('travel')
                                  | data['site_id'].str.contains('tourist'))

        data['is_weather_site'] = (data['site_id'].str.contains('weather'))

        data['is_game_site'] = (data['site_id'].str.contains('game'))

        data['is_music_site'] = (data['site_id'].str.contains('music')
                                 | data['site_id'].str.contains('hip-hop')
                                 | data['site_id'].str.contains('song')  # ?
                                 | data['site_id'].str.contains('guitar'))

        data['is_cook_site'] = (data['site_id'].str.contains('cook')
                                | data['site_id'].str.contains('diet')
                                | data['site_id'].str.contains('fastfood')
                                | data['site_id'].str.contains('cupcake')  # javacupcake.com
                                | data['site_id'].str.contains('grandbaby-cakes')
                                | data['site_id'].str.contains('browneyedbaker')  # baker?
                                | data['site_id'].str.contains('baking')
                                | data['site_id'].str.contains('lovebakesgoodcakes')
                                | data['site_id'].str.contains('bakery'))

        data['is_health_site'] = (data['site_id'].str.contains('health')
                                  | data['site_id'].str.contains('medical'))

        data['is_zodiac_site'] = (data['site_id'].str.contains('astrology')
                                  | data['site_id'].str.contains('zodiac'))

        return data
