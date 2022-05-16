import pandas as pd
import numpy as np
import json
import lightgbm as lgbm
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, accuracy_score

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


def main():
    with open("config.json") as f:
        config_dict = json.load(f)
        model_path = config_dict["model_path"]

    col_names = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation",
                 "relationship",
                 "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "class"]
    df = pd.read_csv("adult.csv", sep=",", header=None, names=col_names, index_col=False)

    train = df.head(int(len(df) * 0.7))
    test = df.tail(len(df) - len(train))
    resp_var = "class"
    X_train = train.drop(resp_var, axis=1)
    y_train = train[resp_var]
    X_test = test.drop(resp_var, axis=1)
    y_test = test[resp_var]

    # One-hot encoding
    final_cols = []
    categorical_feats = ["workclass", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]
    to_drop = ["education"]
    transformers = []
    for column in X_train.columns:
        name = column
        trans = "passthrough"
        if column in categorical_feats:
            trans = OneHotEncoder()
            name = f"{column}_class"
        elif column in to_drop:
            trans = "drop"

        transformers.append((name, trans, [f"{column}"]))

        if trans != "drop":
            final_cols.append(column)
    ct = ColumnTransformer(transformers, remainder="passthrough")
    ct.fit(X_train)
    # Encoder for the labels
    le = LabelEncoder()
    le.fit(y_train)

    X_train_trans = ct.transform(X_train)
    X_test_trans = ct.transform(X_test)
    y_train_trans = le.transform(y_train)
    y_test_trans = le.transform(y_test)

    parameters = {
        "n_estimators": np.geomspace(100, 10000, num=3, dtype=int),
        "num_leaves": np.geomspace(32, 256, num=4, dtype=int),
        "learning_rate": np.geomspace(1e-3, 1e-1, num=3)
    }
    CV_classifier = GridSearchCV(lgbm.LGBMClassifier(n_jobs=16), parameters, verbose=3, scoring="accuracy")
    CV_classifier.fit(X_train_trans, y_train_trans)

    print("Grid search ended, best params: ")
    print(CV_classifier.best_params_)
    forest = lgbm.LGBMClassifier(**CV_classifier.best_params_)
    forest.fit(X_train_trans, y_train_trans)

    print("Accuracy on test set: ")
    print(accuracy_score(y_test_trans, forest.predict(X_test_trans)))

    print("Saving model to:")
    print(model_path)

    forest.booster_.save_model(model_path)


if __name__ == '__main__':
    main()
