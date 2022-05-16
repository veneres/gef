import json
import pandas as pd
import lightgbm as lgbm
import shap
import pickle

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


def main():
    with open("config.json") as f:
        config_dict = json.load(f)
        model_path = config_dict["model_path"]
        shap_values_out = config_dict["shap_values_out"]
        shap_explainer_out = config_dict["shap_explainer_out"]

    forest = lgbm.Booster(model_file=model_path)

    forest.params["objective"] = "binary"

    col_names = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation",
                 "relationship",
                 "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "class"]
    df = pd.read_csv("adult.csv", sep=",", header=None, names=col_names, index_col=False)

    train = df.head(int(len(df) * 0.7))
    resp_var = "class"
    X_train = train.drop(resp_var, axis=1)
    y_train = train[resp_var]

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

    final_cols = ct.get_feature_names_out().copy()
    final_cols[14] = "MS-Married"
    final_cols[47] = "CapitalGain"
    final_cols[11] = "EducationNum"
    final_cols[0] = "Age"

    X_train_trans = ct.transform(X_train)

    shap_explainer = shap.Explainer(forest, feature_names=final_cols)
    shap_values = shap_explainer(X_train_trans.toarray())

    with open(shap_values_out, 'wb') as f:
        pickle.dump(shap_values, f)
    with open(shap_explainer_out, 'wb') as f:
        pickle.dump(shap_explainer, f)


if __name__ == '__main__':
    main()
