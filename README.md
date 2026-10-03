
# Iris Sepal Length Regression: OLS vs. Ridge vs. Lasso
<img width="768" height="1376" alt="iris data info image multi,ridge" src="https://github.com/user-attachments/assets/ed1021b9-7b6c-4b40-907d-b5176a8bcc41" />

A comparative machine learning project analyzing regularized regression techniques on the classic Iris dataset. The primary objective is to predict **Sepal Length (cm)** using flower measurements (*sepal width*, *petal length*, and *petal width*) through Ordinary Least Squares (OLS) Linear Regression, Ridge ($L_2$), and Lasso ($L_1$) regression algorithms.

---

## Key Features

* **Leak-Free Preprocessing Pipelines:** Integrates feature standardization (`StandardScaler`) and estimators within `sklearn.pipeline.make_pipeline` to prevent data leakage across split boundaries and cross-validation folds.
* **Dynamic Hyperparameter Tuning:** Replaces manual alpha selection with `RidgeCV` and `LassoCV` to dynamically discover optimal penalty parameters ($\alpha$) across a logarithmic search space.
* **Visual Analytics Dashboard:** Generates a 4-panel Seaborn and Matplotlib visualization saved as `iris_regression_dashboard.png`:
* **Feature Correlation Heatmap:** Identifies collinearity among sepal and petal features.
* **Standardized Coefficients:** Visualizes parameter shrinkage effects across estimators.
* **Actual vs. Predicted Plot:** Evaluates prediction accuracy against an ideal linear reference fit.
* **5-Fold CV $R^2$ Distribution:** Measures model stability and variance across shuffled validation folds.



---

## Project Structure

```text
.
├── multiple_ridge_lasso_iris.py   # Main Python workflow script
├── iris_regression_dashboard.png  # Generated 2x2 analytical visual dashboard
├── README.md                      # Project documentation
└── requirements.txt               # Required Python packages

```

---

## Installation & Setup

1. **Clone the repository:**
```bash
git clone https://github.com/your-username/iris-regression-regularization.git
cd iris-regression-regularization

```


2. **Create and activate a virtual environment (optional):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

```


3. **Install dependencies:**
```bash
pip install pandas numpy scikit-learn matplotlib seaborn

```



---

## Usage

Execute the complete training, evaluation, and plotting pipeline:

```bash
python multiple_ridge_lasso_iris.py

```

Upon execution, the script outputs performance metrics (MSE, MAE, $R^2$) and optimal regularization alphas to the terminal, and exports `iris_regression_dashboard.png` to the project directory.

---

## Methodology & Model Overview

1. **Feature Standardization:** Features are standardized ($z = \frac{x - \mu}{\sigma}$) prior to training, ensuring regularization penalties apply uniformly across variables regardless of native units.
2. **Regression Models:**
* **Linear Regression (OLS):** Minimizes residual sum of squares without regularization constraints.
* **Ridge Regression ($L_2$):** Incorporates penalty term $\alpha \sum \beta_i^2$ to shrink regression coefficients and mitigate multi-collinearity.
* **Lasso Regression ($L_1$):** Incorporates penalty term $\alpha \sum \vert{}\beta_i\vert{}$ enabling feature selection by shrinking non-critical weights strictly to zero.


3. **Cross-Validation:** Evaluated using 5-fold shuffled cross-validation (`KFold(n_splits=5, shuffle=True)`) to ensure generalizability across different data splits.

---

## Output Metrics

| Model | MSE | MAE | $R^2$ Score | Best $\alpha$ |
| --- | --- | --- | --- | --- |
| **Linear** | Evaluated on test set | Evaluated on test set | Evaluated on test set | N/A |
| **Ridge** | Evaluated on test set | Evaluated on test set | Evaluated on test set | Tuned via CV |
| **Lasso** | Evaluated on test set | Evaluated on test set | Evaluated on test set | Tuned via CV |

---

## Author 
Md Anwarul karim 
MS in DataScience
