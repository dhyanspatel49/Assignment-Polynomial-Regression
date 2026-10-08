import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_validate
import warnings

warnings.filterwarnings("ignore")

def optimize_polynomial_regression(X_train, y_train, max_degree):
    best_degree = 1
    best_score = -np.inf
    best_model = None

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    print(f"Testing degrees 1 through {max_degree}...")
    for degree in range(1, max_degree + 1):
        model = Pipeline([
            ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
            ('scaler', StandardScaler()),
            ('regressor', RidgeCV(alphas=np.logspace(-3, 3, 10)))
        ])
        
        cv_results = cross_validate(model, X_train, y_train, cv=kf, scoring=['r2', 'neg_mean_squared_error'])
        
        mean_r2 = np.mean(cv_results['test_r2'])
        mean_mse = -np.mean(cv_results['test_neg_mean_squared_error'])
        
        print(f"Degree {degree:>2} | Mean R2: {mean_r2:.4f} | Mean MSE: {mean_mse:.4f}")
        
        if mean_r2 > best_score:
            best_score = mean_r2
            best_degree = degree
            best_model = model

    print(f"\n--> Selected Best Degree: {best_degree} (CV R2: {best_score:.4f})")
    
    best_model.fit(X_train, y_train)
    return best_model, best_degree

def process_dataset(train_file, test_file, pred_file, max_degree, feature_cols):
    print(f"\n{'='*40}\nProcessing {train_file}\n{'='*40}")
    
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    
    X_train = train_df[feature_cols]
    y_train = train_df['y']
    X_test = test_df[feature_cols]
    
    model, best_degree = optimize_polynomial_regression(X_train, y_train, max_degree=max_degree)
    
    predictions = model.predict(X_test)
    
    output_df = pd.DataFrame({'y': predictions})
    output_df.to_csv(pred_file, index=False)
    
    print(f"Predictions saved to {pred_file}")

if __name__ == "__main__":
    
    features_var1 = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']
    process_dataset(
        train_file="BT2024075_train_var1.csv",
        test_file="BT2024075_test_var1.csv",
        pred_file="BT2024075_pred_var1.csv",
        max_degree=10,
        feature_cols=features_var1
    )
    
    features_var2 = ['x1', 'x2', 'x3']
    process_dataset(
        train_file="BT2024075_train_var2.csv",
        test_file="BT2024075_test_var2.csv",
        pred_file="BT2024075_pred_var2.csv",
        max_degree=20,
        feature_cols=features_var2
    )
