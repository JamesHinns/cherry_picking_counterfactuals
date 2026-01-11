# So I'd like to put the config.py into the cli.py

# But let's call it:
# dice_experiments.py

# I put hashing.py, runner.py  - inside utils/
# compute_metrics_from_records is in compared_cfs.py, you must now also pass exp_dir as an os.path to the function
# def compute_metrics_from_records(df: pd.DataFrame,
#                                  records: List[Dict[str, Any]],
#                                  to_explain: pd.DataFrame,
#                                  exp_dir: os.path) -> pd.DataFrame:

# The experiments.py file isn't needed.
# So things like build_feature_sets 
# Should be done manually in the calling code, not like this as it's dataset specefic.
# Same with to_explain.
# The experiments should also be done there, so all this code isn't needed.