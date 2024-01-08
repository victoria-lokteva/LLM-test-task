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
