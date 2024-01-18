
example0 ="""
Question:
"On what time of day are the most clicks made?"
Answer example:
>>>>>>>
To answer this question, we create column 'time of day'. We suppose, that morning corresponds to time period
 from 6 am to 12 am, afternoon corresponds to time period from 12 am to 6 pm, evening corresponds to time period from
 6 pm to 12 pm, night corresponds to time period from 12 pm to 6 am. Example of python code:
 
# Convert the 'time' column to datetime format
df['time'] = pd.to_datetime(df['time'], format='%H:%M')

# Define time periods
morning_period = (6, 12)
afternoon_period = (12, 18)
evening_period = (18, 24)
night_period = (0, 6)

# Function to determine time of day
def get_time_of_day(hour):
    if morning_period[0] <= hour < morning_period[1]:
        return 'morning'
    elif afternoon_period[0] <= hour < afternoon_period[1]:
        return 'afternoon'
    elif evening_period[0] <= hour < evening_period[1]:
        return 'evening'
    else:
        return 'night'

# Extract the hour from the 'time' column and apply the function to create 'time of day' column
df['hour'] = df['time'].dt.hour
df['time of day'] = df['hour'].apply(get_time_of_day)

# Drop the intermediate 'hour' column if not needed
df = df.drop(columns=['hour'])
 
 
 We should perform group by operation on the column 'time of day' and count the amount of clicks.

# Perform groupby operation on 'time of day' and count the 'fclick' tags
click_counts = df[df['tag'] == 'fclick'].groupby('time of day')['tag'].count().reset_index()

# Rename the count column
click_counts = click_counts.rename(columns={'tag': 'click_count'})

Now we can find a maximum value in the column click_count:
max_clicks_time_of_day = click_counts.loc[click_counts['click_count'].idxmax(), 'time of day']
<<<<<<<
"""

# 1
example1 ="""
Question:
Build the regression model to predict the price of houses.
Target is "SalePrice", features list is ["country", "state", "flour", "year", "area", "Alley"], 
categorical features list is ["country", "state"]
All data in dataset is preprocessed properly.

Answer example:
<<<<<<<
import joblib
import lightgbm as lgb
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import log_loss, mean_squared_error

train = pd.read_feather('data/train.feather')
test = pd.read_feather('data/test.feather')

target = "SalePrice"
features = ["country", "state", "flour", "year", "area", "Alley"]
categorical_features = ["country", "state"]

x = train.loc[:, features]
y = train[target]

model = LGBMRegressor()

model.fit(x, y, categorical_feature=categorical_features)
          
joblib.dump(model, 'lgbm.pkl')

test["prediction"] = model.predict(test.loc[:, features])

print('MSE: ', mean_squared_error(test[target], test["prediction"]))

<<<<<<<
"""

# 2
example2 ="""
Question:
Build the classification model to predict the price of houses.
Target is "SalePrice", features list is ["country", "state", "flour", "year", "area", "Alley"], 
categorical features list is ["country", "state"]
All data in dataset is preprocessed properly.

Answer example:
<<<<<<<
import joblib
import lightgbm as lgb
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import log_loss, mean_squared_error

train = pd.read_feather('data/train.feather')
test = pd.read_feather('data/test.feather')

target = "SalePrice"
features = ["country", "state", "flour", "year", "area", "Alley"]
categorical_features = ["country", "state"]

x = train.loc[:, features]
y = train[target]

model = LGBMClassifier()

model.fit(x, y, categorical_feature=categorical_features)

joblib.dump(model, 'lgbm.pkl')

test["prediction"] = model.predict_proba(test.loc[:, features])[:, 1]

print('LogLoss: ', log_loss(test[target], test["prediction"]))

<<<<<<<
"""

# 3
example3 ="""
Question:
Build regression model to predict price of houses.
Features list is ["flour", "year", "area", "country", "state", "Alley"], 
categorical features list is ["state", "country"], target is "SalePrice".
All data in dataset is preprocessed properly.

Answer example:
<<<<<<<
import joblib
import lightgbm as lgb
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import log_loss, mean_squared_error

train = pd.read_feather('data/train.feather')
test = pd.read_feather('data/test.feather')

target = "SalePrice"
features = ["country", "state", "flour", "year", "area", "Alley"]
categorical_features = ["country", "state"]

x = train.loc[:, features]
y = train[target]

model = LGBMRegressor()

model.fit(x, y, categorical_feature=categorical_features)

joblib.dump(model, 'lgbm.pkl')

test["prediction"] = model.predict(test.loc[:, features])

print('MSE: ', mean_squared_error(test[target], test["prediction"]))

<<<<<<<
"""

# 4
example4 = """
Question:
Build classification model to predict price of houses.
Features list is ["flour", "year", "area", "country", "state", "Alley"], 
categorical features list is ["state", "country"], target is "SalePrice".
All data in dataset is preprocessed properly.

Answer example:
<<<<<<<
import joblib
import lightgbm as lgb
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import log_loss, mean_squared_error

train = pd.read_feather('data/train.feather')
test = pd.read_feather('data/test.feather')

target = "SalePrice"
features = ["country", "state", "flour", "year", "area", "Alley"]
categorical_features = ["country", "state"]

x = train.loc[:, features]
y = train[target]

model = LGBMClassifier()

model.fit(x, y, categorical_feature=categorical_features)

joblib.dump(model, 'lgbm.pkl')

test["prediction"] = model.predict_proba(test.loc[:, features])[:, 1]

print('LogLoss: ', log_loss(test[target], test["prediction"]))

<<<<<<<
"""

# 5
example5 ="""
Question:
Build regression model.
Features list is ["x", "z"], categorical features list is [], target is "y".

Answer example:
<<<<<<<
import joblib
import lightgbm as lgb
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import log_loss, mean_squared_error

train = pd.read_feather('data/train.feather')
test = pd.read_feather('data/test.feather')

target = "y"
features = ["x", "z"]
categorical_features = []

x = train.loc[:, features]
y = train[target]

model = LGBMRegressor()

model.fit(x, y, categorical_feature=categorical_features)

joblib.dump(model, 'lgbm.pkl')

test["prediction"] = model.predict(test.loc[:, features])

print('MSE: ', mean_squared_error(test[target], test["prediction"]))

<<<<<<<
"""


# 6
example6 ="""
Question:
Build classification model.
Features list is ["x", "z"], categorical features list is [], target is "y".

Answer example:
<<<<<<<
import joblib
import lightgbm as lgb
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import log_loss, mean_squared_error

train = pd.read_feather('data/train.feather')
test = pd.read_feather('data/test.feather')

target = "y"
features = ["x", "z"]
categorical_features = []

x = train.loc[:, features]
y = train[target]

model = LGBMClassifier()

model.fit(x, y, categorical_feature=categorical_features)

joblib.dump(model, 'lgbm.pkl')

test["prediction"] = model.predict_proba(test.loc[:, features])[:, 1]

print('LogLoss: ', log_loss(test[target], test["prediction"]))

<<<<<<<
"""


# 7
example7 ="""
Question:
Build a model. It is a classification task.
Features list is ["feature1", "feature2", "feature3", "feature4"], 
categorical features list is ["feature2", "feature4"], target is "y".

Answer example:
<<<<<<<
import joblib
import lightgbm as lgb
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import log_loss, mean_squared_error

train = pd.read_feather('data/train.feather')
test = pd.read_feather('data/test.feather')

target = "y"
features = ["feature1", "feature2", "feature3", "feature4"]
categorical_features = ["feature2", "feature4"]

x = train.loc[:, features]
y = train[target]

model = LGBMClassifier()

model.fit(x, y, categorical_feature=categorical_features)

joblib.dump(model, 'lgbm.pkl')

test["prediction"] = model.predict_proba(test.loc[:, features])[:, 1]

print('LogLoss: ', log_loss(test[target], test["prediction"]))

<<<<<<<
"""

# 8
example8 ="""
Question:
Classify "y".
Features list is ["feature1", "feature2", "feature3", "feature4"], 
categorical features list is ["feature2", "feature4"], target is "y".

Answer example:
<<<<<<<
import joblib
import lightgbm as lgb
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import log_loss, mean_squared_error

train = pd.read_feather('data/train.feather')
test = pd.read_feather('data/test.feather')

target = "y"
features = ["feature1", "feature2", "feature3", "feature4"]
categorical_features = ["feature2", "feature4"]

x = train.loc[:, features]
y = train[target]

model = LGBMClassifier()

model.fit(x, y, categorical_feature=categorical_features)

joblib.dump(model, 'lgbm.pkl')

test["prediction"] = model.predict_proba(test.loc[:, features])[:, 1]

print('LogLoss: ', log_loss(test[target], test["prediction"]))

<<<<<<<
"""


example9 ="""
Question:
Write a code that classify "y".
Features are ["feature1", "feature2", "feature3", "feature4"], 
categorical features are ["feature2", "feature4"], and target is "y".

Answer example:
<<<<<<<
import joblib
import lightgbm as lgb
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import log_loss, mean_squared_error

train = pd.read_feather('data/train.feather')
test = pd.read_feather('data/test.feather')

target = "y"
features = ["feature1", "feature2", "feature3", "feature4"]
categorical_features = ["feature2", "feature4"]

x = train.loc[:, features]
y = train[target]

model = LGBMClassifier()

model.fit(x, y, categorical_feature=categorical_features)

joblib.dump(model, 'lgbm.pkl')

test["prediction"] = model.predict_proba(test.loc[:, features])[:, 1]

print('LogLoss: ', log_loss(test[target], test["prediction"]))

<<<<<<<
"""

# 10
"""
Question:
Hello!
I need to build the model that predict the probability of y.
I have dataset "../data/mldataset.csv"
Features are ["feature1", "feature2", "feature3", "feature4"], 
categorical features are ["feature2", "feature4"], and target is "y".

Answer example:
<<<<<<<
import joblib
import lightgbm as lgb
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import log_loss, mean_squared_error

train = pd.read_feather('data/train.feather')
test = pd.read_feather('data/test.feather')

target = "y"
features = ["feature1", "feature2", "feature3", "feature4"]
categorical_features = ["feature2", "feature4"]

x = train.loc[:, features]
y = train[target]

model = LGBMClassifier()

model.fit(x, y, categorical_feature=categorical_features)

joblib.dump(model, 'lgbm.pkl')

test["prediction"] = model.predict_proba(test.loc[:, features])[:, 1]

print('LogLoss: ', log_loss(test[target], test["prediction"]))

<<<<<<<
"""

# 11
example11 = """
Question:
I have th task. Need to built the model that predict the probability of event.
I have dataset "../data/clickdataset.csv"
Features are ["feature1", "feature2", "feature3", "feature4"], 
categorical features are ["feature2", "feature4"], and target is "event".

Answer example:
<<<<<<<
import joblib
import lightgbm as lgb
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import log_loss, mean_squared_error

train = pd.read_feather('data/train.feather')
test = pd.read_feather('data/test.feather')

target = "event"
features = ["feature1", "feature2", "feature3", "feature4"]
categorical_features = ["feature2", "feature4"]

x = train.loc[:, features]
y = train[target]

model = LGBMClassifier()

model.fit(x, y, categorical_feature=categorical_features)

joblib.dump(model, 'lgbm.pkl')

test["prediction"] = model.predict_proba(test.loc[:, features])[:, 1]

print('LogLoss: ', log_loss(test[target], test["prediction"]))

<<<<<<<
"""


# 13
example13 ="""
Question:
My dataset consist of the following features ["feature1", "feature2", "feature3", "feature4"]
Features are ["feature1", "feature2", "feature3", "feature4"], 
categorical features are ["feature2", "feature4"], and target is "click".
I Need to built the model that predict the probability of "click".

Answer example:
<<<<<<<
import joblib
import lightgbm as lgb
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import log_loss, mean_squared_error

train = pd.read_feather('data/train.feather')
test = pd.read_feather('data/test.feather')

target = "click"
features = ["feature1", "feature2", "feature3", "feature4"]
categorical_features = ["feature2", "feature4"]

x = train.loc[:, features]
y = train[target]

model = LGBMClassifier()

model.fit(x, y, categorical_feature=categorical_features)

joblib.dump(model, 'lgbm.pkl')

test["prediction"] = model.predict_proba(test.loc[:, features])[:, 1]

print('LogLoss: ', log_loss(test[target], test["prediction"]))

<<<<<<<
"""