from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR


def build_model(config):
    benchmark_type = config.get("benchmark_type", "xgboost").lower()

    if benchmark_type == "xgboost":
        model = XGBRegressor(
            n_estimators=config.get("n_estimators", 300),
            max_depth=config.get("max_depth", 4),
            learning_rate=config.get("learning_rate", 0.01),
            subsample=config.get("subsample", 1.0),
            colsample_bytree=config.get("colsample_bytree", 1.0),
            random_state=config.get("random_state", 42),
            n_jobs=config.get("n_jobs", 1),
        )

    elif benchmark_type == "random_forest":
        model = RandomForestRegressor(
            n_estimators=config.get("n_estimators", 300),
            max_depth=config.get("max_depth", 10),
            min_samples_split=config.get("min_samples_split", 2),
            min_samples_leaf=config.get("min_samples_leaf", 1),
            max_features=config.get("max_features", "sqrt"),
            random_state=config.get("random_state", 42),
            n_jobs=config.get("n_jobs", 1),
        )

    elif benchmark_type == "svr":
        model = SVR(
            kernel=config.get("kernel", "rbf"),
            C=config.get("C", 1.0),
            gamma=config.get("gamma", "scale"),
            epsilon=config.get("epsilon", 0.1)
        )

    else:
        raise ValueError(f"Unsupported benchmark model type: {benchmark_type}")

    return model
