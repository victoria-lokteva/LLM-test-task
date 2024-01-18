import pandas as pd

df = pd.read_csv('../data/dataset.csv')

df = df[df['tag'] == 'fclick']

device_clicks = df['hardware'].value_counts()

device_clicks = device_clicks.sort_values(ascending=False)

print(device_clicks)

