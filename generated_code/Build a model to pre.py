import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from sklearn.metrics import log_loss
import joblib

df = pd.read_csv('../data/dataset.csv')

# Convert 'click' column to binary
df['click'] = df['click'].astype(int)

# Convert 'tag' column to binary where 'fclick' is 1 and others are 0
df['tag'] = df['tag'].apply(lambda x: 1 if x == 'fclick' else 0)

# Label encode categorical features
le = LabelEncoder()
categorical_features = ['osName', 'model', 'hardware', 'site_id', 'domain', 'site_category', 'week_day']
for feature in categorical_features:
    df[feature] = le.fit_transform(df[feature])

features = ['fc_imp_chk', 'fc_time_chk', 'utmtr', 'mm_dma', 'osName', 'model', 'hardware', 'site_id', 'domain', 'site_category', 'year', 'month', 'hour', 'week_day']
target = 'click'
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LGBMClassifier()
model.fit(X_train, y_train, categorical_feature=categorical_features)

y_pred = model.predict_proba(X_test)[:, 1]
print('LogLoss: ', log_loss(y_test, y_pred))

joblib.dump(model, 'lgbm.pkl')
