// Data for Ordinary Least Squares (OLS) Regression algorithm
export const olsRegressionData = {
  name: 'Ordinary Least Squares (OLS)',
  category: 'Supervised Learning',
  description: 'Classic linear regression minimizing sum of squared residuals',
  badges: [
    { label: 'Regression', color: 'blue' },
    { label: 'Linear Model', color: 'green' },
    { label: 'Interpretable', color: 'purple' },
    { label: 'Foundation', color: 'orange' }
  ],
  
  overview: {
    whatIs: {
      title: 'What is Ordinary Least Squares (OLS)?',
      description: 'Ordinary Least Squares is the foundational method for linear regression. It finds the line (or hyperplane) that minimizes the sum of squared differences between observed and predicted values. OLS provides the Best Linear Unbiased Estimator (BLUE) under the Gauss-Markov assumptions, making it the gold standard for linear regression when assumptions are met. It has a closed-form analytical solution, making it fast and exact.',
      highlight: 'OLS is the cornerstone of regression analysis - simple, interpretable, and optimal when assumptions hold. All other regression methods are variations trying to address OLS\'s limitations!'
    },
    
    whenToUse: {
      title: 'When to Use OLS',
      perfectFor: [
        'Linear relationships between features and target',
        'Clean data with no outliers',
        'Features are not highly correlated',
        'Fewer features than samples (p < n)',
        'Assumptions are met (linearity, independence, homoscedasticity)',
        'Need maximum interpretability',
        'Want statistical inference (p-values, confidence intervals)',
        'Baseline model for comparison'
      ],
      avoidWhen: [
        'Multicollinearity present (use Ridge)',
        'Need feature selection (use Lasso)',
        'Data has outliers (use Robust Regression)',
        'More features than samples (use regularization)',
        'Non-linear relationships (use polynomial or non-linear models)',
        'High-dimensional data (regularization needed)',
        'Heteroscedasticity present (weighted least squares better)'
      ]
    },
    
    useCases: {
      title: 'Real-World Use Cases',
      cases: [
        {
          icon: '📚',
          title: 'Academic Research & Hypothesis Testing',
          description: 'Test relationships between variables with statistical rigor. OLS provides p-values, confidence intervals, and R² for hypothesis testing. Standard in social sciences, economics, and medical research for causal inference.'
        },
        {
          icon: '📊',
          title: 'Baseline Model Development',
          description: 'Start every ML project with OLS to understand linear relationships. Provides interpretable benchmark before trying complex models. If OLS works well, simpler is better!'
        },
        {
          icon: '💰',
          title: 'Simple Financial Forecasting',
          description: 'Quick revenue predictions, trend analysis, and budget forecasting. When relationships are linear and data is clean, OLS is fast and interpretable for stakeholders.'
        },
        {
          icon: '🏠',
          title: 'Real Estate Appraisal',
          description: 'Estimate property values from size, location, age. OLS coefficients directly interpretable as "price per square foot" making results easy to explain to clients.'
        },
        {
          icon: '📈',
          title: 'Business Analytics & Reporting',
          description: 'Sales forecasting, customer lifetime value, marketing ROI analysis. OLS provides clear, interpretable insights for business decisions with straightforward assumptions.'
        },
        {
          icon: '🔬',
          title: 'Experimental Data Analysis',
          description: 'Analyze controlled experiments where assumptions likely hold. Laboratory conditions minimize outliers and confounding. Perfect for establishing cause-effect relationships.'
        }
      ]
    },
    
    prosAndCons: {
      title: 'Advantages & Limitations',
      pros: [
        'Closed-Form Solution: Exact analytical solution, no iterations needed',
        'Highly Interpretable: Coefficients have clear meaning (unit change in Y per unit X)',
        'Statistical Inference: Provides p-values, confidence intervals, hypothesis tests',
        'Computationally Efficient: Fast even with large datasets',
        'BLUE Property: Best Linear Unbiased Estimator under Gauss-Markov assumptions',
        'Well-Understood: Extensive theoretical foundation and software support',
        'No Hyperparameters: No tuning needed, deterministic results',
        'Diagnostic Tools: Rich ecosystem for checking assumptions'
      ],
      cons: [
        'Sensitive to Outliers: Single outlier can drastically affect fit',
        'Multicollinearity Issues: Unstable with highly correlated features',
        'No Feature Selection: Keeps all features regardless of importance',
        'Overfitting Risk: With many features relative to samples',
        'Assumption Dependent: Performance degrades when assumptions violated',
        'Linear Only: Cannot capture non-linear relationships',
        'No Regularization: Prone to overfitting in high dimensions',
        'Fails with p ≥ n: Cannot solve when features ≥ samples'
      ]
    },
    
    stepByStep: {
      title: 'Step-by-Step Algorithm',
      steps: [
        {
          title: 'Set Up Problem',
          description: 'Define objective: minimize ||y - Xβ||² where y is target, X is feature matrix, β is coefficient vector. This is equivalent to minimizing sum of squared residuals.'
        },
        {
          title: 'Form Normal Equations',
          description: 'Take derivative of objective with respect to β and set to zero: ∂/∂β ||y - Xβ||² = 0. This gives normal equations: XᵀXβ = Xᵀy.'
        },
        {
          title: 'Check Matrix Invertibility',
          description: 'Verify XᵀX is invertible. Requires: (1) n > p (more samples than features), (2) features are linearly independent. If not invertible, use regularization instead.'
        },
        {
          title: 'Solve for Coefficients',
          description: 'Compute β = (XᵀX)⁻¹Xᵀy. This is the closed-form OLS solution. Use Cholesky decomposition or QR decomposition for numerical stability.'
        },
        {
          title: 'Compute Fitted Values',
          description: 'Calculate predicted values: ŷ = Xβ. These are the model\'s predictions on the training data.'
        },
        {
          title: 'Calculate Residuals',
          description: 'Compute residuals: e = y - ŷ. These are prediction errors. Check residual plots to validate assumptions.'
        },
        {
          title: 'Estimate Error Variance',
          description: 'Calculate σ² = ||e||²/(n-p) where n is samples, p is features. This is the unbiased estimate of error variance.'
        },
        {
          title: 'Compute Standard Errors',
          description: 'Calculate SE(β) = σ√diag((XᵀX)⁻¹). Standard errors enable confidence intervals and hypothesis tests.'
        },
        {
          title: 'Statistical Inference',
          description: 'Compute t-statistics: t = β/SE(β) and p-values. Test H₀: βⱼ = 0. Also compute R², adjusted R², F-statistic for model fit.'
        }
      ]
    }
  },
  
  math: {
    objectiveFunction: {
      title: 'OLS Objective Function',
      formula: 'minimize: RSS(β) = ||y - Xβ||² = Σᵢ(yᵢ - xᵢᵀβ)²',
      parameters: [
        'y: Target variable (n × 1 vector)',
        'X: Feature matrix (n × p matrix)',
        'β: Coefficient vector (p × 1 vector)',
        'RSS: Residual Sum of Squares',
        'n: Number of samples',
        'p: Number of features',
        'No regularization term (unlike Ridge/Lasso)',
        'Also called: Sum of Squared Errors (SSE)'
      ]
    },
    additionalContent: [
      {
        title: 'Closed-Form Solution',
        formula: 'β̂ = (XᵀX)⁻¹Xᵀy',
        description: 'Exact analytical solution. Unique when XᵀX is invertible (requires n > p and linearly independent features). This is the Best Linear Unbiased Estimator (BLUE).'
      },
      {
        title: 'Normal Equations',
        formula: 'XᵀXβ = Xᵀy',
        description: 'System of linear equations derived by setting gradient to zero. Solving gives OLS solution. Can be solved efficiently with Cholesky or QR decomposition.'
      },
      {
        title: 'Fitted Values',
        formula: 'ŷ = Xβ̂ = X(XᵀX)⁻¹Xᵀy = Hy',
        description: 'H = X(XᵀX)⁻¹Xᵀ is the "hat matrix" (puts hat on y). Projects y onto column space of X. Diagonal elements hᵢᵢ measure leverage.'
      },
      {
        title: 'Residuals',
        formula: 'e = y - ŷ = y - Hy = (I - H)y',
        description: 'Prediction errors. Key assumption: e ~ N(0, σ²I) for inference. Check residual plots: should be random, constant variance, normally distributed.'
      },
      {
        title: 'Variance of Coefficients',
        formula: 'Var(β̂) = σ²(XᵀX)⁻¹',
        description: 'Covariance matrix of coefficient estimates. Diagonal elements are variances: Var(β̂ⱼ) = σ²[(XᵀX)⁻¹]ⱼⱼ. Standard errors: SE(β̂ⱼ) = √Var(β̂ⱼ).'
      },
      {
        title: 'R-squared',
        formula: 'R² = 1 - RSS/TSS = 1 - Σ(yᵢ-ŷᵢ)²/Σ(yᵢ-ȳ)²',
        description: 'Proportion of variance explained. 0 ≤ R² ≤ 1. R²=1 means perfect fit. Adjusted R²: R²ₐ = 1 - (1-R²)(n-1)/(n-p-1) penalizes additional features.'
      },
      {
        title: 'F-Statistic',
        formula: 'F = (TSS-RSS)/p / (RSS/(n-p-1))',
        description: 'Tests H₀: all βⱼ = 0. F ~ F(p, n-p-1) under null. Large F (small p-value) rejects null, indicates model is useful.'
      },
      {
        title: 't-Statistic for Individual Coefficients',
        formula: 'tⱼ = β̂ⱼ / SE(β̂ⱼ)',
        description: 'Tests H₀: βⱼ = 0. Under null, tⱼ ~ t(n-p-1). |t| > 2 roughly indicates significance at 5% level (depends on df).'
      }
    ],
    visualization: {
      title: 'Geometric Interpretation',
      items: [
        {
          title: 'Projection',
          color: '#3b82f6',
          description: 'OLS projects target vector y onto the column space of X (space spanned by feature vectors). Fitted values ŷ are the projection.'
        },
        {
          title: 'Residuals',
          color: '#ef4444',
          description: 'Residual vector e is orthogonal to column space of X. This means Xᵀe = 0 (residuals uncorrelated with features at optimum).'
        },
        {
          title: 'Least Squares',
          color: '#8b5cf6',
          description: 'OLS finds ŷ that minimizes ||e||². Geometrically, finds closest point in column space of X to y (shortest distance = smallest error).'
        }
      ],
      insight: 'OLS is a projection problem: projecting y onto the space spanned by features. The solution is unique when feature vectors are linearly independent. Residuals are orthogonal to feature space, meaning we cannot reduce error further while staying linear.'
    }
  },
  
  code: {
    examples: [
      {
        title: 'Python (scikit-learn)',
        code: `from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import pandas as pd

# Prepare data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Method 1: Basic OLS
ols = LinearRegression(
    fit_intercept=True,  # Include intercept
    copy_X=True          # Copy X to avoid modifying original
)

ols.fit(X_train, y_train)
y_pred = ols.predict(X_test)

# Model parameters
print(f"Intercept: {ols.intercept_:.4f}")
print(f"Coefficients: {ols.coef_}")

# Evaluate
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\\nTest Set Performance:")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")

# Prediction intervals (approximate)
residuals = y_train - ols.predict(X_train)
std_error = np.std(residuals)
confidence_level = 1.96  # 95% confidence

y_lower = y_pred - confidence_level * std_error
y_upper = y_pred + confidence_level * std_error

print(f"\\nPrediction interval width: ±{confidence_level * std_error:.4f}")

# Method 2: With statsmodels (more statistical output)
import statsmodels.api as sm

# Add constant for intercept
X_train_const = sm.add_constant(X_train)
X_test_const = sm.add_constant(X_test)

# Fit OLS
ols_sm = sm.OLS(y_train, X_train_const)
results = ols_sm.fit()

# Comprehensive summary
print("\\n" + "="*70)
print(results.summary())
print("="*70)

# Individual statistics
print(f"\\nR-squared: {results.rsquared:.4f}")
print(f"Adjusted R-squared: {results.rsquared_adj:.4f}")
print(f"F-statistic: {results.fvalue:.4f}, p-value: {results.f_pvalue:.4e}")
print(f"AIC: {results.aic:.4f}")
print(f"BIC: {results.bic:.4f}")

# Coefficient details
coef_summary = pd.DataFrame({
    'Coefficient': results.params,
    'Std Error': results.bse,
    't-value': results.tvalues,
    'p-value': results.pvalues,
    '95% CI Lower': results.conf_int()[0],
    '95% CI Upper': results.conf_int()[1]
})
print("\\nCoefficient Summary:")
print(coef_summary)

# Diagnostic plots
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Residuals vs Fitted
axes[0, 0].scatter(results.fittedvalues, results.resid, alpha=0.5)
axes[0, 0].axhline(0, color='red', linestyle='--')
axes[0, 0].set_xlabel('Fitted values')
axes[0, 0].set_ylabel('Residuals')
axes[0, 0].set_title('Residuals vs Fitted')

# Q-Q plot
from scipy import stats
stats.probplot(results.resid, dist="norm", plot=axes[0, 1])
axes[0, 1].set_title('Normal Q-Q Plot')

# Scale-Location
standardized_resid = results.resid / np.std(results.resid)
axes[1, 0].scatter(results.fittedvalues, np.sqrt(np.abs(standardized_resid)), alpha=0.5)
axes[1, 0].set_xlabel('Fitted values')
axes[1, 0].set_ylabel('√|Standardized Residuals|')
axes[1, 0].set_title('Scale-Location')

# Residuals histogram
axes[1, 1].hist(results.resid, bins=30, edgecolor='black')
axes[1, 1].set_xlabel('Residuals')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Residual Distribution')

plt.tight_layout()
plt.show()`
      },
      {
        title: 'R (built-in)',
        code: `# Prepare data
set.seed(42)
train_index <- sample(1:nrow(data), 0.8 * nrow(data))
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

# Method 1: Basic OLS
ols_model <- lm(y ~ ., data = train_data)

# Comprehensive summary
summary(ols_model)

# Key statistics
cat("R-squared:", summary(ols_model)$r.squared, "\\n")
cat("Adjusted R-squared:", summary(ols_model)$adj.r.squared, "\\n")
cat("Residual Std Error:", summary(ols_model)$sigma, "\\n")

# Coefficients with confidence intervals
coef_summary <- data.frame(
  Estimate = coef(ols_model),
  Std_Error = summary(ols_model)$coefficients[, "Std. Error"],
  t_value = summary(ols_model)$coefficients[, "t value"],
  p_value = summary(ols_model)$coefficients[, "Pr(>|t|)"]
)
print(coef_summary)

# 95% confidence intervals
confint(ols_model)

# Predictions
predictions <- predict(ols_model, newdata = test_data)

# Prediction intervals
pred_intervals <- predict(ols_model, newdata = test_data, 
                         interval = "prediction", level = 0.95)
print(head(pred_intervals))

# Evaluate
actual <- test_data$y
mse <- mean((actual - predictions)^2)
rmse <- sqrt(mse)
mae <- mean(abs(actual - predictions))
r2 <- 1 - sum((actual - predictions)^2) / sum((actual - mean(actual))^2)

cat("\\nTest Set Performance:\\n")
cat("MSE:", mse, "\\n")
cat("RMSE:", rmse, "\\n")
cat("MAE:", mae, "\\n")
cat("R²:", r2, "\\n")

# Diagnostic plots
par(mfrow = c(2, 2))
plot(ols_model)

# Additional diagnostics
library(car)

# Variance Inflation Factor (multicollinearity)
vif_values <- vif(ols_model)
print("VIF values (>10 indicates multicollinearity):")
print(vif_values)

# Durbin-Watson test (autocorrelation)
dw_test <- durbinWatsonTest(ols_model)
print(dw_test)

# Breusch-Pagan test (heteroscedasticity)
bp_test <- ncvTest(ols_model)
print(bp_test)`
      },
      {
        title: 'Python (From Scratch)',
        code: `import numpy as np

class OLSRegressionScratch:
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = 0
        self.se_ = None  # Standard errors
        self.t_stats_ = None  # t-statistics
        self.p_values_ = None  # p-values
        
    def fit(self, X, y):
        """
        Fit OLS: β = (X^T X)^(-1) X^T y
        """
        n_samples, n_features = X.shape
        
        # Add intercept column if needed
        if self.fit_intercept:
            X_design = np.column_stack([np.ones(n_samples), X])
        else:
            X_design = X.copy()
        
        # Compute OLS solution
        XtX = X_design.T @ X_design
        Xty = X_design.T @ y
        
        # Check if invertible
        if np.linalg.cond(XtX) > 1e10:
            print("Warning: Design matrix is ill-conditioned. Consider regularization.")
        
        # Solve normal equations
        beta = np.linalg.solve(XtX, Xty)
        
        # Extract intercept and coefficients
        if self.fit_intercept:
            self.intercept_ = beta[0]
            self.coef_ = beta[1:]
        else:
            self.intercept_ = 0
            self.coef_ = beta
        
        # Compute statistics
        self._compute_statistics(X, y, X_design, beta)
        
        return self
    
    def _compute_statistics(self, X, y, X_design, beta):
        """Compute standard errors, t-stats, p-values"""
        n = len(y)
        p = X_design.shape[1]
        
        # Fitted values and residuals
        y_pred = X_design @ beta
        residuals = y - y_pred
        
        # Residual sum of squares
        rss = np.sum(residuals**2)
        
        # Estimate error variance
        sigma2 = rss / (n - p)
        
        # Variance-covariance matrix of coefficients
        XtX_inv = np.linalg.inv(X_design.T @ X_design)
        var_beta = sigma2 * XtX_inv
        
        # Standard errors
        se_all = np.sqrt(np.diag(var_beta))
        
        # t-statistics
        t_stats_all = beta / se_all
        
        # p-values (two-tailed test)
        from scipy import stats
        p_values_all = 2 * (1 - stats.t.cdf(np.abs(t_stats_all), n - p))
        
        # Store (excluding intercept if present)
        if self.fit_intercept:
            self.se_ = se_all[1:]
            self.t_stats_ = t_stats_all[1:]
            self.p_values_ = p_values_all[1:]
        else:
            self.se_ = se_all
            self.t_stats_ = t_stats_all
            self.p_values_ = p_values_all
    
    def predict(self, X):
        return X @ self.coef_ + self.intercept_
    
    def summary(self):
        """Print coefficient summary"""
        print("\\nOLS Regression Results")
        print("=" * 60)
        for i, (coef, se, t, p) in enumerate(zip(
            self.coef_, self.se_, self.t_stats_, self.p_values_
        )):
            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
            print(f"Feature {i}: β={coef:.4f}, SE={se:.4f}, t={t:.4f}, p={p:.4f} {sig}")
        print("=" * 60)
        print("Significance: *** p<0.001, ** p<0.01, * p<0.05")

# Example usage
np.random.seed(42)
n_samples, n_features = 100, 3
X = np.random.randn(n_samples, n_features)
true_coef = np.array([2.5, -1.5, 3.0])
y = X @ true_coef + np.random.randn(n_samples) * 0.5

# Fit OLS
ols = OLSRegressionScratch(fit_intercept=True)
ols.fit(X, y)

print(f"True coefficients: {true_coef}")
print(f"Estimated coefficients: {ols.coef_}")
print(f"Intercept: {ols.intercept_:.4f}")

# Summary
ols.summary()

# Predictions
y_pred = ols.predict(X)
r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
print(f"\\nR²: {r2:.4f}")`
      }
    ]
  },
  
  preprocessing: {
    critical: {
      title: 'OLS Assumptions to Check',
      description: 'OLS requires several assumptions for optimal performance and valid statistical inference. Always check these!',
      code: `# 1. Linearity: Plot y vs X\n# 2. Independence: Check residual autocorrelation\n# 3. Homoscedasticity: Plot residuals vs fitted\n# 4. Normality: Q-Q plot of residuals\n# 5. No multicollinearity: Check VIF`,
      why: 'When assumptions are violated: (1) Linearity → Use polynomial/non-linear models, (2) Heteroscedasticity → Use weighted least squares, (3) Multicollinearity → Use Ridge, (4) Outliers → Use robust regression'
    },
    stepsTitle: 'Recommended Preprocessing Steps',
    steps: [
      {
        title: 'Handle Missing Values',
        description: 'OLS requires complete data. Impute or remove missing values.',
        code: `from sklearn.impute import SimpleImputer\n\nimputer = SimpleImputer(strategy='mean')\nX_imputed = imputer.fit_transform(X)`
      },
      {
        title: 'Check for Outliers',
        description: 'OLS is very sensitive to outliers. Visualize and consider removal or robust methods.',
        code: `import matplotlib.pyplot as plt\n\nplt.boxplot(X, vert=False)\nplt.show()\n\n# Cook's distance for influential points\nfrom statsmodels.stats.outliers_influence import OLSInfluence\ninfluence = OLSInfluence(results)\ncooks_d = influence.cooks_distance[0]\n# Points with Cook's D > 1 are influential`
      },
      {
        title: 'Encode Categorical Variables',
        description: 'Convert categorical features to numerical.',
        code: `from sklearn.preprocessing import OneHotEncoder\n\nencoder = OneHotEncoder(drop='first', sparse=False)\nX_encoded = encoder.fit_transform(X_categorical)`
      },
      {
        title: 'Check Multicollinearity',
        description: 'High correlation between features causes unstable coefficients.',
        code: `from statsmodels.stats.outliers_influence import variance_inflation_factor\n\nvif_data = pd.DataFrame()\nvif_data["Feature"] = X.columns\nvif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]\nprint(vif_data)\n# VIF > 10 indicates problematic multicollinearity`
      },
      {
        title: 'Feature Scaling (Optional)',
        description: 'Not required for OLS correctness, but helps with interpretation and numerical stability.',
        code: `from sklearn.preprocessing import StandardScaler\n\nscaler = StandardScaler()\nX_scaled = scaler.fit_transform(X)\n# Makes coefficients comparable across features`
      }
    ],
    completePipeline: {
      title: 'Complete Pipeline with Validation',
      code: `from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# Preprocessing pipeline
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('ols', LinearRegression())
])

# Fit
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# For full statistical output, use statsmodels
X_train_processed = pipeline[:-1].transform(X_train)
X_train_const = sm.add_constant(X_train_processed)
ols_sm = sm.OLS(y_train, X_train_const).fit()

# Check assumptions
print("Checking OLS Assumptions:")
print("="*50)

# 1. Linearity (informal check via R²)
print(f"R²: {ols_sm.rsquared:.4f}")

# 2. Heteroscedasticity (Breusch-Pagan test)
from statsmodels.stats.diagnostic import het_breuschpagan
bp_test = het_breuschpagan(ols_sm.resid, X_train_const)
print(f"Breusch-Pagan p-value: {bp_test[1]:.4f}")
print("  → p>0.05: Homoscedasticity assumption met")

# 3. Normality of residuals (Jarque-Bera test)
from statsmodels.stats.stattools import jarque_bera
jb_test = jarque_bera(ols_sm.resid)
print(f"Jarque-Bera p-value: {jb_test[1]:.4f}")
print("  → p>0.05: Normality assumption met")

# 4. Autocorrelation (Durbin-Watson)
from statsmodels.stats.stattools import durbin_watson
dw_stat = durbin_watson(ols_sm.resid)
print(f"Durbin-Watson: {dw_stat:.4f}")
print("  → ~2.0: No autocorrelation")`
    },
    mistakes: {
      title: 'Common Mistakes',
      items: [
        'Not checking assumptions before using OLS',
        'Ignoring multicollinearity (VIF > 10)',
        'Using OLS with outliers present',
        'Applying OLS when p ≥ n (need regularization)',
        'Not examining residual plots',
        'Treating non-linear relationships as linear',
        'Comparing coefficients without standardizing features'
      ]
    }
  },
  
  tips: {
    hyperparameterTuning: {
      title: 'Model Selection and Validation',
      sections: [
        {
          title: 'Feature Selection Methods',
          points: [
            'Forward Selection: Start with no features, add one at a time',
            'Backward Elimination: Start with all, remove insignificant ones',
            'Stepwise Selection: Combination of forward and backward',
            'Best Subset Selection: Try all possible combinations (expensive)',
            'Use AIC/BIC for model comparison: lower is better',
            'Cross-validation: More reliable than p-values for feature selection'
          ]
        },
        {
          title: 'Model Diagnostics',
          points: [
            'R²: Proportion of variance explained (higher better)',
            'Adjusted R²: Penalizes adding features (better for comparison)',
            'AIC: Akaike Information Criterion (lower better)',
            'BIC: Bayesian Information Criterion (lower better, stricter than AIC)',
            'RMSE: Root Mean Squared Error (lower better)',
            'Plot residuals: Should be random, constant variance'
          ]
        },
        {
          title: 'When to Use Alternatives',
          points: [
            'Multicollinearity (VIF>10): Use Ridge Regression',
            'Need feature selection: Use Lasso or Elastic Net',
            'Outliers present: Use Robust Regression (Huber, RANSAC)',
            'p ≥ n: Must use regularization (Ridge/Lasso)',
            'Non-linear: Use polynomial features or non-linear models',
            'Heteroscedasticity: Use Weighted Least Squares (WLS)'
          ]
        }
      ]
    },
    bestPractices: {
      dos: [
        'Always check assumptions before trusting results',
        'Plot residuals vs fitted values',
        'Check Q-Q plot for normality',
        'Calculate and examine VIF for multicollinearity',
        'Use cross-validation for honest performance estimates',
        'Report confidence intervals, not just point estimates',
        'Test on held-out test set',
        'Start with OLS as baseline before trying complex models'
      ],
      donts: [
        'Do not ignore assumption violations',
        'Do not use with severe multicollinearity',
        'Do not trust p-values when assumptions violated',
        'Do not use when p ≥ n (mathematically impossible)',
        'Do not ignore outliers',
        'Do not extrapolate far beyond training data range',
        'Do not compare R² across different datasets',
        'Do not add features just to increase R² (overfitting)'
      ]
    },
    advancedTechniques: [
      {
        title: 'Weighted Least Squares (WLS)',
        description: 'Handle heteroscedasticity by weighting observations',
        code: `import statsmodels.api as sm\n\n# Estimate weights (inverse of variance)\nresid_squared = results.resid**2\nweights = 1 / resid_squared\n\n# Fit WLS\nwls_model = sm.WLS(y, X, weights=weights)\nwls_results = wls_model.fit()\nprint(wls_results.summary())`
      },
      {
        title: 'Polynomial Regression',
        description: 'Capture non-linear relationships while staying in OLS framework',
        code: `from sklearn.preprocessing import PolynomialFeatures\n\n# Create polynomial features\npoly = PolynomialFeatures(degree=2, include_bias=False)\nX_poly = poly.fit_transform(X)\n\n# Fit OLS on polynomial features\nols_poly = LinearRegression()\nols_poly.fit(X_poly, y)\n\n# Now can model non-linear relationships`
      },
      {
        title: 'Bootstrap Confidence Intervals',
        description: 'Non-parametric confidence intervals without normality assumption',
        code: `from sklearn.utils import resample\n\nbootstrap_coefs = []\nfor i in range(1000):\n    X_boot, y_boot = resample(X_train, y_train, random_state=i)\n    ols_boot = LinearRegression()\n    ols_boot.fit(X_boot, y_boot)\n    bootstrap_coefs.append(ols_boot.coef_)\n\nbootstrap_coefs = np.array(bootstrap_coefs)\n\n# 95% confidence intervals\nci_lower = np.percentile(bootstrap_coefs, 2.5, axis=0)\nci_upper = np.percentile(bootstrap_coefs, 97.5, axis=0)\nprint("Bootstrap 95% CI:", list(zip(ci_lower, ci_upper)))`
      },
      {
        title: 'Cross-Validation for OLS',
        description: 'Honest performance estimate',
        code: `from sklearn.model_selection import cross_val_score\n\nols = LinearRegression()\nscores = cross_val_score(ols, X, y, cv=5, scoring='r2')\n\nprint(f"CV R² scores: {scores}")
print(f"Mean CV R²: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")`
      }
    ],
    performance: {
      title: 'Performance Optimization',
      tips: [
        'Closed-form solution: OLS is very fast, O(np²) time complexity',
        'Use QR decomposition for numerical stability',
        'Cholesky decomposition: Fastest when XᵀX well-conditioned',
        'For very large n: Use stochastic gradient descent approximation',
        'For sparse X: Use sparse matrix operations',
        'Cache XᵀX if fitting with different y multiple times',
        'Parallel processing: sklearn automatically uses BLAS/LAPACK',
        'For huge datasets: Consider sampling or online learning'
      ]
    },
    debugging: {
      title: 'Debugging Common Issues',
      issues: [
        {
          problem: 'Problem: Singular matrix error',
          solution: 'Perfect multicollinearity. Check for duplicate features, perfect correlations, or p ≥ n. Use Ridge regression or remove redundant features.'
        },
        {
          problem: 'Problem: Very large/small coefficients',
          solution: 'Features have very different scales or multicollinearity. Standardize features or check VIF. Large coefficients with large standard errors indicate instability.'
        },
        {
          problem: 'Problem: High R² on train, low on test',
          solution: 'Overfitting. Too many features relative to samples. Use regularization (Ridge/Lasso) or reduce features. Add more training data if possible.'
        },
        {
          problem: 'Problem: Non-significant coefficients but high R²',
          solution: 'Multicollinearity. Individual coefficients unreliable but model predictions OK. Check VIF. Use Ridge or combine correlated features.'
        },
        {
          problem: 'Problem: Residuals show clear pattern',
          solution: 'Violated linearity assumption. Try polynomial features, transformations (log, sqrt), or non-linear models. Pattern indicates systematic error.'
        },
        {
          problem: 'Problem: Heteroscedasticity in residuals',
          solution: 'Non-constant variance. Use Weighted Least Squares (WLS), transform y (log/sqrt), or use robust standard errors for inference.'
        },
        {
          problem: 'Problem: Some predictions are negative when y should be positive',
          solution: 'OLS can predict outside valid range. Try log-transformation of y, use GLM (Poisson/Gamma), or apply constraints in post-processing.'
        }
      ]
    }
  }
};