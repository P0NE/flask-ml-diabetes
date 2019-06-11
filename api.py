from flask import Flask, request, jsonify
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

app = Flask(__name__)

# import data from csv
df = pd.read_csv("data/pima-data.csv")

# delete column skin for coordonnance
del df["skin"]

# change true to 1 and false to 0 for boolean column values
diabetes_map = {True: 1, False: 0}
df["diabetes"] = df["diabetes"].map(diabetes_map)

# spliting the data => 70% train, 30% test
feature_col_names = ['num_preg', 'glucose_conc', 'diastolic_bp',
                     'thickness', 'insulin', 'bmi', 'diab_pred', 'age']
predicted_class_names = ['diabetes']
X = df[feature_col_names].values
y = df[predicted_class_names].values
split_test_size = 0.30
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=split_test_size, random_state=42)
print("{0:0.2f}% in training set".format((len(X_train)/len(df.index)) * 100))
print("{0:0.2f}% in test set".format((len(X_test)/len(df.index)) * 100))

# impute rows missing with the mean
fill_0 = SimpleImputer(missing_values=0, strategy='mean', verbose=0)
X_train = fill_0.fit_transform(X_train)
X_test = fill_0.fit_transform(X_test)

# training model with logistic regression
lr_model = LogisticRegression(
    class_weight='balanced', C=0.300, random_state=42)
lr_model.fit(X_train, y_train.ravel())


@app.route('/')
def index():
    return "Welcome to Diabetes diagnostic Machine Learnig API"


@app.route('/testingModel')
def testing_modele():
    lr_predict_test = lr_model.predict(X_test)
    return jsonify(accuracy="{0:.4f}".format(metrics.accuracy_score(y_test, lr_predict_test)))


if __name__ == "__main__":
    HOST = os.environ.get('SERVER_HOST', 'localhost')
    PORT = int(os.environ.get('SERVER_PORT', '5000'))
    app.run(HOST, PORT, debug=True)
