import os
import warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # hide INFO/WARN/ERROR from TF C++ side
os.environ["AUTOGRAPH_VERBOSITY"] = "0"
from abc import ABC, abstractmethod

import pandas as pd
import numpy as np

import dice_ml
from dice_ml import Data, Model

from carla import Data as CarlaData, MLModel as CarlaMLModel
from carla.recourse_methods import GrowingSpheres, Wachter 

class CounterfactualAlgorithm(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def generate_counterfactuals(self):
        pass

class DiCE(CounterfactualAlgorithm):

    def __init__(self, data_df, model, method,
                 cont_features, disc_features):
        
        self.method = method

        # Think this should be handled differently
        if cont_features is None:
            cont_features = [c for c in data_df.columns if data_df[c].dtype != 'object' and c != "Label"]
        if disc_features is None:
            disc_features = [c for c in data_df.columns if data_df[c].dtype == 'object' and c != "Label"]

        data_interface = Data(
            dataframe=data_df,
            continuous_features=cont_features,
            categorical_features=disc_features,
            outcome_name="Label"
        )

        model_interface = Model(model=model, backend="sklearn") # static backend for current rf example

        self.explainer = dice_ml.Dice(
            data_interface=data_interface,
            model_interface=model_interface,
            method=self.method
        )

    def generate_counterfactuals(self, to_explain, cf_seed, **kwargs):
        # DiCE discrete and continous throws warnings for dtype issues
        # We silence them here for simplicity
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Setting an item of incompatible dtype is deprecated",
                category=FutureWarning,
                module=r"dice_ml\.explainer_interfaces\.dice_random"
            )
            # This one is also a strange but seemingly uneeded warning
            # https://stackoverflow.com/questions/68292862/performancewarning-dataframe-is-highly-fragmented-this-is-usually-the-result-o
            warnings.filterwarnings(
                "ignore",
                category=pd.errors.PerformanceWarning,
                module=r"dice_ml\.explainer_interfaces\.dice_KD",
            )
            # kdtree and genetic don't support random seed
            if self.method in ["kdtree", "genetic"]:
                cf_result = self.explainer.generate_counterfactuals(
                    query_instances=to_explain,
                    desired_class="opposite",
                    total_CFs=1,
                    **kwargs
                )
            else:
                cf_result = self.explainer.generate_counterfactuals(
                    query_instances=to_explain,
                    random_seed=cf_seed,
                    desired_class="opposite",
                    total_CFs=1,
                    **kwargs
                )

        # If we fail to find a counterfactual, void the results
        # We may need to revisit this, but it wouldn't be fair to return incomplete lists
        # of explanations, and if a counterfactual cannot be found where it could be before, 
        # that could also be a red flag?
        # maybe also we should just allow individual instances to have all 0s?
        # arbritray choice of what the values are, so long as we have a standard for "not found",
        # this could still effect plausibility though
        if any(ex.final_cfs_df is None for ex in cf_result.cf_examples_list):
            n_fail = sum(ex.final_cfs_df is None for ex in cf_result.cf_examples_list)
            print(f"[DiCE] CF generation failed for {n_fail}/{len(cf_result.cf_examples_list)} instances.")
            return None

        cfs = [example.final_cfs_df.copy() for example in cf_result.cf_examples_list]
        df_cfs = pd.concat(cfs, ignore_index=True)
        df_cfs.index = to_explain.index
        df_cfs = df_cfs[to_explain.columns]
        return df_cfs
    
# ------ CARLA wrappers ------

# We need a number of wrappers to make CARLA work with our setup
# It also has ancient dependencies, which should probably be handled properly
# everything here is a bodge to fix this.

# A key issue is most CARLA methods only accept pytorch models, and need differntiable models
# For this reason we experiment with only logistic regression pytorch models and use:
# Wachter, FACE, GrowingSpheres, Actionable Recourse, Casuaul Recourse