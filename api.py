from flask import Flask, request, jsonify
import os
import pandas as pd
import numpy as np
from werkzeug.exceptions import BadRequest
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


def args_grouping_required(func):
    def wraps(*args, **kwargs):
        if not request.args.get('num_preg'):
            raise BadRequest({"error": "missing.argument.num_preg"})
        if not request.args.get('glucose_conc'):
            raise BadRequest({"error": "missing.argument.glucose_conc"})
        if not request.args.get('diastolic_bp'):
            raise BadRequest({"error": "missing.argument.diastolic_bp"})
        if not request.args.get('thickness'):
            raise BadRequest({"error": "missing.argument.thickness"})
        if not request.args.get('insulin'):
            raise BadRequest({"error": "missing.argument.insulin"})
        if not request.args.get('bmi'):
            raise BadRequest({"error": "missing.argument.bmi"})
        if not request.args.get('diab_pred'):
            raise BadRequest({"error": "missing.argument.diab_pred"})
        if not request.args.get('age'):
            raise BadRequest({"error": "missing.argument.age"})
        return func(*args, **kwargs)
    return wraps


def accuracy_score():
    lr_predict_test = lr_model.predict(X_test)
    return metrics.accuracy_score(y_test, lr_predict_test)


@app.route('/')
def index():
    return "Welcome to Diabetes diagnostic Machine Learnig API"


@app.route('/testingModel')
def testing_modele():
    lr_predict_test = lr_model.predict(X_test)
    return jsonify(accuracy="{0:.4f}".format(metrics.accuracy_score(y_test, lr_predict_test)))


@app.route('/predictDiabete')
@args_grouping_required
def predict_diabete():
    data = [[request.args.get('num_preg'), request.args.get(
        'glucose_conc'), request.args.get('diastolic_bp'), request.args.get('thickness'), request.args.get('insulin'), request.args.get('bmi'), request.args.get('diab_pred'), request.args.get('age')]]
    df = pd.DataFrame(data, columns=['num_preg', 'glucose_conc', 'diastolic_bp',
                                     'thickness', 'insulin', 'bmi', 'diab_pred', 'age'])
    print(df)
    lr_predict_test = lr_model.predict(df)
    print(lr_predict_test)
    res = True if lr_predict_test[0] == 1 else False
    return jsonify(resultat=res, accuracy=accuracy_score())


if __name__ == "__main__":
    HOST = os.environ.get('SERVER_HOST', 'localhost')
    PORT = int(os.environ.get('SERVER_PORT', '5000'))
    app.run(HOST, PORT, debug=True)
