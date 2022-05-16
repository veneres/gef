import pandas as pd
import numpy as np
import json
import lightgbm as lgbm
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import GridSearchCV


def main():
    with open("config.json") as f:
        config_dict = json.load(f)
        model_path = config_dict["model_path"]

    df = pd.read_csv("train.csv", sep=",")
    train = df.head(int(len(df) * 0.7))
    test = df.tail(len(df) - len(train))
    resp_var = "critical_temp"
    X_train = train.drop(resp_var, axis=1)
    y_train = train[resp_var]
    X_test = test.drop(resp_var, axis=1)
    y_test = test[resp_var]
    parameters = {
        "n_estimators": np.geomspace(100, 10000, num=3, dtype=int),
        "num_leaves": np.geomspace(32, 256, num=4, dtype=int),
        "learning_rate": np.geomspace(1e-3, 1e-1, num=3),
        "n_jobs": [40]
    }
    CV_regressor = GridSearchCV(lgbm.LGBMRegressor(), parameters, verbose=3, scoring="neg_root_mean_squared_error")
    CV_regressor.fit(X_train, y_train)

    print("Grid search ended, best params: ")
    print(CV_regressor.best_params_)
    forest = lgbm.LGBMRegressor(**CV_regressor.best_params_)
    forest.fit(X_train, y_train)

    print("RMSE on test set: ")
    print(mean_squared_error(y_test, forest.predict(X_test), squared=False))

    print("Saving model to:")
    print(model_path)

    forest.booster_.save_model(model_path)


if __name__ == '__main__':
    main()
