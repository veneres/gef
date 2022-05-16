import json
import pandas as pd
import lightgbm as lgbm
import shap
import pickle


def main():
    with open("config.json") as f:
        config_dict = json.load(f)
        model_path = config_dict["model_path"]
        shap_values_out = config_dict["shap_values_out"]
        shap_explainer_out = config_dict["shap_explainer_out"]

    forest = lgbm.Booster(model_file=model_path)

    forest.params["objective"] = "regression"

    df = pd.read_csv("train.csv", sep=",")
    train = df.head(int(len(df) * 0.7))
    resp_var = "critical_temp"
    X_train = train.drop(resp_var, axis=1)

    shap_explainer = shap.Explainer(forest)
    shap_values = shap_explainer(X_train)

    with open(shap_values_out, 'wb') as f:
        pickle.dump(shap_values, f)
    with open(shap_explainer_out, 'wb') as f:
        pickle.dump(shap_explainer, f)


if __name__ == '__main__':
    main()
