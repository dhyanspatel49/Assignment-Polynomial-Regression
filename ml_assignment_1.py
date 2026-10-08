import time
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_validate

warnings.filterwarnings("ignore")


def build_model(name, degree):
    if name == "Ridge":
        regressor = RidgeCV(alphas=np.logspace(-3, 3, 10))
    else:
        regressor = LassoCV(n_alphas=50, cv=3, max_iter=5000, tol=1e-3, random_state=42)
    return Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("scaler", StandardScaler()),
        ("regressor", regressor),
    ])


def compare_models(X_train, y_train, max_degree, model_names=("Ridge", "Lasso")):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rows = []
    for name in model_names:
        for degree in range(1, max_degree + 1):
            start = time.time()
            cv = cross_validate(
                build_model(name, degree), X_train, y_train, cv=kf,
                scoring=["r2", "neg_mean_squared_error"], n_jobs=-1
            )
            r2 = np.mean(cv["test_r2"])
            mse = -np.mean(cv["test_neg_mean_squared_error"])
            rows.append({"model": name, "degree": degree, "mean_r2": r2, "mean_mse": mse})
            print(f"{name:<5} | Degree {degree:>2} | Mean R2: {r2:.4f} | Mean MSE: {mse:.4f} | {time.time() - start:.1f}s")
    return pd.DataFrame(rows)


def process_dataset(train_file, test_file, pred_file, results_file, max_degree, feature_cols):
    print(f"\n{'=' * 50}\nProcessing {train_file}\n{'=' * 50}")

    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    X_train = train_df[feature_cols]
    y_train = train_df["y"]
    X_test = test_df[feature_cols]

    results = compare_models(X_train, y_train, max_degree)
    results.to_csv(results_file, index=False)

    print("\nBest degree per model:")
    for name, grp in results.groupby("model"):
        top = grp.loc[grp["mean_r2"].idxmax()]
        print(f"{name:<5} -> Degree {int(top['degree'])} | R2: {top['mean_r2']:.4f} | MSE: {top['mean_mse']:.4f}")

    best = results.loc[results["mean_r2"].idxmax()]
    best_name, best_degree = best["model"], int(best["degree"])
    print(f"\n--> Overall Best: {best_name}, Degree {best_degree} (CV R2: {best['mean_r2']:.4f}, CV MSE: {best['mean_mse']:.4f})")

    final_model = build_model(best_name, best_degree)
    final_model.fit(X_train, y_train)
    regressor = final_model.named_steps["regressor"]
    print(f"Selected alpha: {regressor.alpha_:.6f}")
    if best_name == "Lasso":
        n_nonzero = int(np.sum(regressor.coef_ != 0))
        print(f"Non-zero coefficients: {n_nonzero} of {regressor.coef_.shape[0]}")

    predictions = final_model.predict(X_test)
    pd.DataFrame({"y": predictions}).to_csv(pred_file, index=False)
    print(f"Predictions saved to {pred_file}")


if __name__ == "__main__":
    process_dataset(
        train_file="BT2024075_train_var1.csv",
        test_file="BT2024075_test_var1.csv",
        pred_file="BT2024075_pred_var1.csv",
        results_file="BT2024075_cv_results_var1.csv",
        max_degree=8,
        feature_cols=["x1", "x2", "x3", "x4", "x5", "x6"],
    )

    process_dataset(
        train_file="BT2024075_train_var2.csv",
        test_file="BT2024075_test_var2.csv",
        pred_file="BT2024075_pred_var2.csv",
        results_file="BT2024075_cv_results_var2.csv",
        max_degree=20,
        feature_cols=["x1", "x2", "x3"],
    )
