import pandas as pd
from typing import Tuple


def preprocessing(interview_x_path: str, interview_y_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """"""
    interview_x = pd.read_csv(interview_x_path)
    interview_y = pd.read_csv(interview_y_path)
    # уберем дупликаты
    interview_x = interview_x.drop_duplicates()
    interview_y = interview_y.drop_duplicates()
    # заполним пропуски для категориальных фич новым значением
    for col in ['osName', 'model', 'hardware']:
        interview_x.loc[interview_x[col].isna(), col] = 'unknown'

    # конвертируем строки в даты
    interview_x['reg_time'] = pd.to_datetime(interview_x['reg_time'], format='%Y-%m-%d %H:%M:%S')

    # из site_id можно извлечь много информации
    # уберем www, www1
    interview_x['site_id'] = interview_x['site_id'].map(
        lambda x: x.replace("www1.", "").replace("www.", ""))
    # Добавим фичу домен
    interview_x['domain'] = interview_x['site_id'].map(lambda x: x.split('.')[-1])
    interview_x = extract_catgories(interview_x)

    # значения dma, которые встречаются менее 5 раз, объединим в одну категорию -1 ("другое")
    rare_dma = [dma for dma, count in interview_x['mm_dma'].value_counts().items() if count < 5]
    interview_x.loc[interview_x['mm_dma'].isin(rare_dma), 'mm_dma'] = -1

    # Аналогично поступим с site_id, model
    rare_site = [site for site, count in interview_x['site_id'].value_counts().items() if count < 5]
    interview_x.loc[interview_x['site_id'].isin(rare_site), 'site_id'] = "other"

    rare_model = [model for model, count in interview_x['model'].value_counts().items() if count < 5]
    interview_x.loc[interview_x['model'].isin(rare_model), 'model'] = "other"

    interview = interview_x.merge(interview_y, how='inner', on='uid')

    return interview_x, interview_y, interview


def extract_catgories(interview_x: pd.DataFrame) -> pd.DataFrame:
    """Добавим категории (спорт/здоровье и т.д.), к которым относится сайт, по ключевым словам из названия"""
    interview_x['sport_cat'] = (interview_x['site_id'].str.contains('sport')
                                | interview_x['site_id'].str.contains('baseball')
                                | interview_x['site_id'].str.contains('football')
                                | interview_x['site_id'].str.contains('soccer')
                                | interview_x['site_id'].str.contains('basketball')
                                | interview_x['site_id'].str.contains('hockey')
                                | interview_x['site_id'].str.contains('moneyball')
                                | interview_x['site_id'].str.contains('rotowire')
                                | interview_x['site_id'].str.contains('mmafighting')
                                | interview_x['site_id'].str.contains('mmamania')
                                | interview_x['site_id'].str.contains('nbaanalysis')
                                | interview_x['site_id'].str.contains('nba-trade-rumors')
                                | interview_x['site_id'].str.contains('nfltraderumors')
                                | interview_x['site_id'].str.contains('nfldraftdiamonds')
                                | interview_x['site_id'].str.contains('detroitbadboys.com')  # ?
                                | interview_x['site_id'].str.contains('volleyball'))

    interview_x['travel_cat'] = (interview_x['site_id'].str.contains('travel')
                                 | interview_x['site_id'].str.contains('tourist'))
    interview_x['weather_cat'] = (interview_x['site_id'].str.contains('weather'))
    interview_x['game_cat'] = (interview_x['site_id'].str.contains('game'))
    interview_x['music_cat'] = (interview_x['site_id'].str.contains('music')
                                | interview_x['site_id'].str.contains('hip-hop')
                                | interview_x['site_id'].str.contains('song')  # ?
                                | interview_x['site_id'].str.contains('guitar'))

    interview_x['cook_cat'] = (interview_x['site_id'].str.contains('cook')
                               | interview_x['site_id'].str.contains('diet')
                               | interview_x['site_id'].str.contains('fastfood')
                               | interview_x['site_id'].str.contains('cupcake')  # javacupcake.com
                               | interview_x['site_id'].str.contains('grandbaby-cakes')
                               | interview_x['site_id'].str.contains('browneyedbaker')  # baker?
                               | interview_x['site_id'].str.contains('baking')
                               | interview_x['site_id'].str.contains('lovebakesgoodcakes')
                               | interview_x['site_id'].str.contains('bakery'))

    interview_x['health_cat'] = (interview_x['site_id'].str.contains('health')
                                 | interview_x['site_id'].str.contains('medical'))

    interview_x['zodiac_cat'] = (interview_x['site_id'].str.contains('astrology')
                                 | interview_x['site_id'].str.contains('zodiac'))

    return interview_x
