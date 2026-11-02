import shap
# shap_analysis.py
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import os
tf.experimental.numpy.experimental_enable_numpy_behavior()
def run_shap_analysis(model, Xp_train, Xm_train, Xp_test, Xm_test, y_test,
                      price_features, macro_features, year,
                      all_shap_values):

    background_size = min(200, len(Xp_train))
    test_sample_size = min(50, len(Xm_test))

    if background_size == 0 or test_sample_size == 0:
        print(f" Skipping SHAP for {year} due to insufficient data.")
        return

    background_indices = np.random.choice(len(Xp_train), background_size, replace=False)
    test_indices = np.random.choice(len(Xp_test), test_sample_size, replace=False)

    background_price = tf.constant(Xp_train[background_indices], dtype=tf.float32)
    background_macro = tf.constant(Xm_train[background_indices], dtype=tf.float32)

    test_price = tf.constant(Xp_test[test_indices], dtype=tf.float32)
    test_macro = tf.constant(Xm_test[test_indices], dtype=tf.float32)

    # Run SHAP explainer
    explainer = shap.GradientExplainer(model, [background_price, background_macro])
    shap_values_list = explainer.shap_values([test_price, test_macro])

    shap_price = shap_values_list[0]
    shap_macro = shap_values_list[1]

    mean_price = np.abs(shap_price).mean(axis=(0, 1)).flatten()
    mean_macro = np.abs(shap_macro).mean(axis=(0, 1)).flatten()

    current_df = pd.DataFrame({
        'feature': price_features + macro_features,
        'shap_value': np.concatenate([mean_price, mean_macro])
    }).set_index('feature')

    all_shap_values.append(current_df)

    # Optionally: save raw SHAP values per year
    os.makedirs("output/shap/", exist_ok=True)
    current_df.to_csv(f"output/shap/shap_values_{year}.csv")


def summarize_shap_results(all_shap_values, results, years, top_n=30):
    if not all_shap_values:
        print("\n No SHAP values were calculated.")
        return

    successful_years = [r[0] for r in results if r[0] in years]
    shap_series = [df["shap_value"] for df in all_shap_values]
    shap_combined = pd.concat(shap_series, axis=1)

    shap_combined.columns = [f"SHAP_{y}" for y in successful_years[:len(shap_combined.columns)]]
    shap_combined["mean_abs_shap"] = shap_combined.mean(axis=1)

    # Sort by mean SHAP importance
    shap_sorted = shap_combined.sort_values(by="mean_abs_shap", ascending=False)

    # Save full and top-N
    os.makedirs("output/shap/", exist_ok=True)
    shap_sorted.to_csv("output/shap/shap_summary_full.csv")

    top_features = shap_sorted.head(top_n)
    top_features.to_csv(f"output/shap/shap_top_{top_n}.csv")

    plt.figure(figsize=(12, 6))
    top_features["mean_abs_shap"].plot(kind="bar", title=f"Top {top_n} Features by Mean SHAP Value")
    plt.ylabel("Mean Absolute SHAP Value")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"output/shap/shap_top_{top_n}_plot.png")
    plt.close()

    print(f"\n Top {top_n} Features by Average SHAP Value Across All Years:")
    print(top_features[["mean_abs_shap"]].round(6))
