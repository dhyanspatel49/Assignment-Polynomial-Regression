# Polynomial Regression Assignment (BT2024075)

Polynomial regression models for two problems:

- **var1:** Power plant steam turbine optimization (6 features, predict `y`)
- **var2:** Subterranean thermal reservoir mapping (3 features, predict `y`)

## Approach

For each dataset, the pipeline is:

1. `PolynomialFeatures` (degree chosen by cross-validation)
2. `StandardScaler`
3. `RidgeCV` (L2 regularization, alpha tuned internally)

The degree is selected using 5-fold cross-validation on the training set (highest mean R²):

| Problem | Degrees searched | Selected degree | CV R² |
|---------|------------------|-----------------|-------|
| var1    | 1-10             | 5               | 0.9512 |
| var2    | 1-20             | 10              | 0.9937 |

See `ml_assignment report.pdf` for the full write-up.

## Files

| File | Description |
|------|-------------|
| `ml_assignment_1.py` | Training, degree selection and prediction code |
| `BT2024075_pred_var1.csv` | Predictions for the var1 test set |
| `BT2024075_pred_var2.csv` | Predictions for the var2 test set |
| `ml_assignment report.pdf` | Project report |

## How to run

1. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn
   ```
2. Place the dataset files in the same folder as the script:
   `BT2024075_train_var1.csv`, `BT2024075_test_var1.csv`, `BT2024075_train_var2.csv`, `BT2024075_test_var2.csv`
3. Run:
   ```bash
   python ml_assignment_1.py
   ```

The script prints the cross-validation R² and MSE for each degree, then writes `BT2024075_pred_var1.csv` and `BT2024075_pred_var2.csv`.
