"""Training pipeline with MLflow tracking and Optuna tuning."""

import argparse
import logging

import mlflow
import mlflow.sklearn
import optuna
import yaml
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from data_loader import load_data, validate_schema
from feature_engineering import ChurnFeatureTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        config: dict = yaml.safe_load(f)
    return config


def objective(trial, X_train, y_train, cv_folds: int) -> float:
    """Optuna objective for XGBoost hyperparameter tuning."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
    }

    model = XGBClassifier(**params, eval_metric="auc", random_state=42)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
    return float(scores.mean())


def train(config_path: str, input_path: str, model_name: str = "xgboost_churn"):
    """Run the full training pipeline."""
    config = load_config(config_path)
    cfg_model = config["model"]
    cfg_feat = config["features"]
    cfg_train = config["training"]
    cfg_mlflow = config["mlflow"]

    # Setup MLflow
    mlflow.set_tracking_uri(cfg_mlflow["tracking_uri"])
    mlflow.set_experiment(cfg_mlflow["experiment_name"])

    # Load data
    df = load_data(input_path)
    validate_schema(df)

    # Split features and target
    feature_cols = cfg_feat["numerical"] + cfg_feat["categorical"]
    X = df[feature_cols]
    y = df[cfg_feat["target"]].astype(int)

    # Split before fitting the transformer. Fitting it on the whole frame
    # leaks the test rows into the scaler means and the encoder classes, so
    # the reported AUC comes out higher than the model would score on data
    # it has never seen.
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg_train["test_size"],
        random_state=cfg_train["random_state"],
        stratify=y,
    )

    transformer = ChurnFeatureTransformer(
        numerical_cols=cfg_feat["numerical"],
        categorical_cols=cfg_feat["categorical"],
    )
    X_train = transformer.fit_transform(X_train_raw)
    X_test = transformer.transform(X_test_raw)

    # Hyperparameter optimization
    if cfg_train.get("optimize", False):
        logger.info("Running Optuna optimization with %d trials", cfg_train["n_trials"])
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: objective(trial, X_train, y_train, cfg_train["cv_folds"]),
            n_trials=cfg_train["n_trials"],
        )
        best_params = study.best_params
        logger.info("Best params: %s", best_params)
    else:
        best_params = cfg_model["params"]

    # Train final model. best_params may already carry eval_metric /
    # early_stopping_rounds when it comes straight from config (optimize:
    # false), so fill in defaults instead of passing them positionally,
    # which would collide with the unpacked dict.
    final_params = dict(best_params)
    final_params.setdefault("eval_metric", "auc")
    final_params.setdefault("random_state", 42)

    with mlflow.start_run():
        model = XGBClassifier(**final_params)

        if final_params.get("early_stopping_rounds"):
            # Early stopping needs rows the model does not fit on. It used to
            # watch the test set, which picks the boosting round that scores
            # best on the same split the AUC below is reported from. Hold out
            # a validation slice of the training rows instead.
            X_fit, X_val, y_fit, y_val = train_test_split(
                X_train,
                y_train,
                test_size=cfg_train.get("validation_size", 0.2),
                random_state=cfg_train["random_state"],
                stratify=y_train,
            )
            model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
        else:
            model.fit(X_train, y_train, verbose=False)

        # Evaluate
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        auc = roc_auc_score(y_test, y_pred_proba)

        logger.info("Test AUC-ROC: %.4f", auc)
        logger.info("\n%s", classification_report(y_test, y_pred))

        # Log to MLflow. We register the transformer and the model together
        # as one pipeline so predict.py always applies the exact same
        # feature transform the model was trained on, instead of the model
        # alone (which would silently score raw, untransformed columns).
        inference_pipeline = Pipeline(
            [
                ("features", transformer),
                ("model", model),
            ]
        )

        mlflow.log_params(final_params)
        mlflow.log_metric("auc_roc", auc)
        mlflow.sklearn.log_model(
            inference_pipeline,
            "model",
            registered_model_name=model_name,
            # skops (the newer default) refuses to deserialize our custom
            # ChurnFeatureTransformer / XGBoost classes as "untrusted
            # types". cloudpickle has no such allowlist and round-trips
            # the pipeline as-is.
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )

        logger.info("Model pipeline registered in MLflow as '%s'", model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/model_config.yaml")
    parser.add_argument("--input", default="data/raw/churn_data.csv")
    parser.add_argument("--model-name", default="xgboost_churn")
    args = parser.parse_args()
    train(args.config, args.input, args.model_name)
