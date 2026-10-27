"""Train and evaluate one model with walk-forward validation.

    python main.py --model transformer --config config/con_transformers.json
    python main.py --model xgboost --config config/ben_xgb.json --commodity copper --seeds 0 1 2

Prints per-year metrics next to the zero-change forecast and saves the daily
forecasts to output/runs/. For the full comparison across models and
commodities use scripts/run_experiments.py and scripts/summarise.py.
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import COMMODITIES, load_commodity
from src.evaluation import diebold_mariano, forecast_metrics, pesaran_timmermann
from src.train_tf import resolve_model_name, walk_forward


def commodity_from_config(config: dict) -> str:
    name = Path(config.get("dataset", "data/gold_fixed_auto.csv")).name
    for key, fname in COMMODITIES.items():
        if fname == name:
            return key
    raise ValueError(f"Cannot tell the commodity from dataset {name!r}; pass --commodity")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="e.g. transformer, lstm_attention, xgboost")
    parser.add_argument("--config", required=True)
    parser.add_argument("--commodity", choices=list(COMMODITIES), help="overrides the dataset in the config")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--years", nargs="+", type=int)
    parser.add_argument("--shap", action="store_true", help="run SHAP on the neural network after training")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    config = json.loads(Path(args.config).read_text())
    model = resolve_model_name(args.model, config)
    commodity = args.commodity or commodity_from_config(config)
    df = load_commodity(commodity)

    runs = []
    kept = []
    for seed in args.seeds:
        preds, models = walk_forward(df, model, config, seed=seed, years=args.years,
                                     verbose=int(args.verbose), keep_models=args.shap)
        runs.append(preds)
        kept.extend(models)
    preds = pd.concat(runs, ignore_index=True)

    out_dir = Path("output/runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{commodity}_{model}.csv"
    preds.to_csv(out_file, index=False)

    ens = preds.groupby(["date", "year"], as_index=False).agg(y_true=("y_true", "first"),
                                                             y_pred=("y_pred", "mean"))
    rows = []
    for year, g in [*ens.groupby("year"), ("all", ens)]:
        m = forecast_metrics(g.y_true, g.y_pred)
        z = forecast_metrics(g.y_true, np.zeros(len(g)))
        _, dm_p = diebold_mariano(g.y_true, g.y_pred, np.zeros(len(g)))
        _, pt_p = pesaran_timmermann(g.y_true, g.y_pred)
        rows.append({
            "year": year,
            "rmse_bp": m["rmse_bp"],
            "zero_rmse_bp": z["rmse_bp"],
            "r2_vs_zero": m["r2_vs_zero"],
            "dir_acc": m["directional_accuracy"],
            "always_up": float((g.y_true[g.y_true != 0] > 0).mean()),
            "dm_p": dm_p,
            "pt_p": pt_p,
        })
    print(f"\n{model} on {commodity}, seeds {args.seeds}")
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print(f"\nDaily forecasts saved to {out_file}")

    if args.shap:
        from src.shap_analysis import run_shap_analysis, summarize_shap_results

        all_shap, results = [], []
        for net, fold in kept:
            if not hasattr(net, "predict") or model in ("xgboost", "random_forest", "svr"):
                print("SHAP is only set up for the neural networks.")
                break
            run_shap_analysis(net, fold.Xp_fit, fold.Xm_fit, fold.Xp_test, fold.Xm_test, fold.y_test,
                              config["price_features"], config["macro_features"], fold.year, all_shap)
            results.append((fold.year,))
        if all_shap:
            summarize_shap_results(all_shap, results, sorted({f.year for _, f in kept}))


if __name__ == "__main__":
    main()
