"""Predict Click Probability:
Can you create a model that tells us how likely it is for someone to click on an online ad?
 Use the data we have and provide an estimate of the probability."""


from lightgbm import LGBMClassifier
import pandas as pd
from preprocessing import preprocessing
from sklearn.model_selection import train_test_split

interview_x_path = "/Users/victorialokteva/Downloads/interviewdata/interview.X.csv"
interview_y_path = "/Users/victorialokteva/Downloads/interviewdata/interview.y.csv"

x, y, _ = preprocessing(interview_x_path, interview_y_path)

train_x, test_x, train_y, test_y = train_test_split(x, y, shuffle=False, test_size=0.1)

LGBMClassifier