import numpy as np
import pandas as pd
import json
import random

from scipy.special import expit
from sklearn.linear_model import LogisticRegression

import dice_ml
from dice_ml import Data, Model

from utils.compare_cfs import count_feature_changes, calc_proximity, calc_sparsity
from utils.load_data import generate_dataset, load_adult

def train_linear_regression(df):
    X = df.drop(columns=["Label"])
    y = df["Label"]

    model = LogisticRegression().fit(X, y)
    print(f"Accuracy: {model.score(X, y):.2f}")

    coefs = model.coef_[0]
    bias = model.intercept_[0]

    for idx, (name, w) in enumerate(zip(X.columns, coefs)):
        print(f"w{idx} ({name}) = {w:.6f}")
    print(f"bias = {bias:.6f}")

    return model

def generate_counterfactuals_dice(model, df, features_to_vary=None, desired_class="opposite",
                                  query_instances=None):

    if features_to_vary is None:
        features_to_vary = [c for c in df.columns if c != "Label"]

    data_interface = Data(
        dataframe=df,
        continuous_features=[c for c in df.columns if df[c].dtype != "object" and c != "Label"],
        outcome_name="Label"
    )
    model_interface = Model(
        model=model,
        backend="sklearn"
    )
    explainer = dice_ml.Dice(
        data_interface,
        model_interface,
        method="random"
    )

    if query_instances is None:
        query_instances = df.drop(columns=['Label'])

    cf_result = explainer.generate_counterfactuals(
        query_instances,
        features_to_vary=features_to_vary,
        desired_class=desired_class,
        total_CFs=1
    )

    all_cfs = []
    for orig_idx, cf_example in zip(df.index, cf_result.cf_examples_list):
        single_cf = cf_example.final_cfs_df.copy()
        single_cf.index = [orig_idx]
        all_cfs.append(single_cf)
    df_cfs = pd.concat(all_cfs)

    return df_cfs

def toy_example():
    df = generate_dataset(500, seed=42, noise=0.0)
    x = df.drop(columns=["Label"])

    model = train_linear_regression(df)

    cf = generate_counterfactuals_dice(model, df).drop(columns=["Label"])
    cf_cherry = generate_counterfactuals_dice(model, df, features_to_vary=["Income"]).drop(columns=["Label"])

    non_cherry_changes = count_feature_changes(cf, x)
    cherry_changes     = count_feature_changes(cf_cherry, x)

    non_cherry_prox = calc_proximity(cf, x)
    cherry_prox     = calc_proximity(cf_cherry, x)
    non_cherry_sp   = calc_sparsity(cf, x)
    cherry_sp       = calc_sparsity(cf_cherry, x)

    print("Non-Cherry:")
    for feat, cnt in non_cherry_changes.items():
        print(f"{feat}: {cnt}")
    print("Proximity:")
    print(non_cherry_prox.mean())
    print("Sparsity:")
    print(non_cherry_sp.mean())

    print("\nCherry:")
    for feat, cnt in cherry_changes.items():
        print(f"{feat}: {cnt}")
    print("Proximity:")
    print(cherry_prox.mean())
    print("Sparsity:")
    print(cherry_sp.mean())

    results = {
        "non_cherry": {
            "changes": non_cherry_changes,
            "proximity": non_cherry_prox.to_dict(orient='list'),
            "sparsity": non_cherry_sp.tolist()
        },
        "cherry": {
            "changes": cherry_changes,
            "proximity": cherry_prox.to_dict(orient='list'),
            "sparsity": cherry_sp.tolist()
        }
    }

    with open("toy_example_results.json", "w") as f:
        json.dump(results, f)

def adult_example(sample_size=500):
    df = load_adult()
    x = df.sample(n=sample_size, random_state=42).drop(columns=["Label"])

    model = train_linear_regression(df)

    cf = generate_counterfactuals_dice(model, df, query_instances=x).drop(columns=["Label"])
    cf.index = x.index

    cherry_pick_cols = ["sex","race","native-country","martial-status"]
    vary_cols = [c for c in df.columns if c not in cherry_pick_cols and c != "Label"]
    cf_cherry = generate_counterfactuals_dice(model, df, features_to_vary=vary_cols,
                                              query_instances=x
                                              ).drop(columns=["Label"])
    cf_cherry.index = x.index

    non_cherry_changes = count_feature_changes(cf, x)
    cherry_changes     = count_feature_changes(cf_cherry, x)

    non_cherry_prox = calc_proximity(cf, x)
    cherry_prox     = calc_proximity(cf_cherry, x)
    non_cherry_sp   = calc_sparsity(cf, x)
    cherry_sp       = calc_sparsity(cf_cherry, x)

    print("Non-Cherry:")
    for feat, cnt in non_cherry_changes.items():
        print(f"{feat}: {cnt}")
    print("Proximity:")
    print(non_cherry_prox.mean())
    print("Sparsity:")
    print(non_cherry_sp.mean())

    print("\nCherry:")
    for feat, cnt in cherry_changes.items():
        print(f"{feat}: {cnt}")
    print("Proximity:")
    print(cherry_prox.mean())
    print("Sparsity:")
    print(cherry_sp.mean())

    results = {
        "non_cherry": {
            "changes": non_cherry_changes,
            "proximity": non_cherry_prox.to_dict(orient='list'),
            "sparsity": non_cherry_sp.tolist()
        },
        "cherry": {
            "changes": cherry_changes,
            "proximity": cherry_prox.to_dict(orient='list'),
            "sparsity": cherry_sp.tolist()
        }
    }

    with open("adult_results.json", "w") as f:
        json.dump(results, f)

def main():
    print("Running toy example...")
    toy_example()

    print("Running adult...")
    adult_example()

if __name__ == "__main__":
    main()