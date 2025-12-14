// Data for Robust Regression algorithms
export const robustRegressionData = {
    name: 'Robust Regression',
    category: 'Supervised Learning',
    description: 'Regression methods resistant to outliers and violations of assumptions',
    badges: [
        { label: 'Regression', color: 'blue' },
        { label: 'Outlier-Resistant', color: 'red' },
        { label: 'Robust Statistics', color: 'orange' },
        { label: 'Non-parametric', color: 'purple' }
    ],

    overview: {
        whatIs: {
            title: 'What is Robust Regression?',
            description: 'Robust Regression is a family of regression techniques designed to be resistant to outliers and violations of standard assumptions (like normally distributed errors). Unlike ordinary least squares (OLS) which minimizes squared errors and is highly sensitive to outliers, robust methods use alternative loss functions or estimation procedures that downweight or ignore outliers. Main methods include Huber Regression, RANSAC, and Theil-Sen Estimator.',
            highlight: 'Think of robust regression as "armor" for your linear model - it protects against bad data points that would otherwise ruin your predictions. Essential when working with real-world messy data!'
        },

        whenToUse: {
            title: 'When to Use Robust Regression',
            perfectFor: [
                'Dataset contains outliers or anomalies',
                'Data quality is questionable or noisy',
                'Violations of normality assumptions',
                'Heavy-tailed error distributions',
                'Contaminated data from multiple sources',
                'Cannot manually clean outliers',
                'Need stable estimates despite bad data',
                'Measurement errors or recording mistakes present'
            ],
            avoidWhen: [
                'Data is clean with no outliers (use OLS - faster)',
                'Outliers are actually important signals to preserve',
                'Very small sample sizes (<50 points)',
                'Need maximum efficiency with perfect data',
                'Computational speed is critical (robust methods slower)',
                'All assumptions of OLS are satisfied'
            ]
        },

        useCases: {
            title: 'Real-World Use Cases',
            cases: [
                {
                    icon: '📊',
                    title: 'Economic & Financial Data',
                    description: 'Economic data often has extreme events (market crashes, recessions). Robust regression provides stable estimates that aren\'t dominated by rare events. Perfect for GDP prediction, inflation modeling with crisis periods.'
                },
                {
                    icon: '🏭',
                    title: 'Industrial Process Control',
                    description: 'Manufacturing sensors can malfunction or give spurious readings. Robust regression ensures quality control models aren\'t thrown off by occasional sensor failures or measurement errors.'
                },
                {
                    icon: '🔬',
                    title: 'Scientific Experiments',
                    description: 'Lab measurements often contain outliers from contamination, instrument errors, or recording mistakes. Robust methods provide reliable parameter estimates despite occasional bad measurements.'
                },
                {
                    icon: '🌍',
                    title: 'Environmental Monitoring',
                    description: 'Environmental sensors in harsh conditions may produce erratic readings. Robust regression filters out sensor failures while maintaining accurate predictions of pollution, temperature, rainfall.'
                },
                {
                    icon: '🏥',
                    title: 'Medical Data Analysis',
                    description: 'Patient data often has recording errors, measurement mistakes, or truly unusual patients. Robust methods provide population-level insights not dominated by outliers.'
                },
                {
                    icon: '🎯',
                    title: 'Computer Vision',
                    description: 'RANSAC is widely used for fitting lines, planes, or transformations to image data containing outlier points from occlusions, shadows, or mismatches.'
                }
            ]
        },

        prosAndCons: {
            title: 'Advantages & Limitations',
            pros: [
                'Outlier Resistance: Not sensitive to extreme values',
                'Stable Estimates: Small changes in data don\'t drastically change results',
                'Assumption Relaxation: Works with heavy-tailed error distributions',
                'Automatic Outlier Handling: No need to manually remove outliers',
                'Better for Real Data: More realistic for messy, real-world datasets',
                'Multiple Methods: Can choose appropriate robust method for problem',
                'Preserves Information: Doesn\'t require deleting data points'
            ],
            cons: [
                'Computational Cost: Slower than OLS',
                'Less Efficient: Slightly higher variance than OLS on clean data',
                'Method Selection: Need to choose appropriate robust method',
                'Convergence Issues: Some methods may not converge',
                'Less Theory: Statistical inference more complex than OLS',
                'Tuning Required: Parameters like loss function threshold need tuning',
                'Masking: May miss meaningful outliers that should be investigated'
            ]
        },

        stepByStep: {
            title: 'Step-by-Step Algorithms',
            steps: [
                {
                    title: 'Huber Regression - Initialize',
                    description: 'Start with initial coefficient estimates (often from OLS). Set epsilon parameter (threshold for considering points as outliers). Typically epsilon = 1.35 for 95% efficiency.'
                },
                {
                    title: 'Huber - Compute Residuals',
                    description: 'Calculate residuals: rᵢ = yᵢ - xᵢᵀβ for each data point. These measure prediction errors.'
                },
                {
                    title: 'Huber - Apply Huber Loss',
                    description: 'Use piecewise loss: L(r) = r²/2 if |r| ≤ ε, else ε|r| - ε²/2. This is quadratic for small errors (like OLS) but linear for large errors (downweights outliers).'
                },
                {
                    title: 'Huber - Update Coefficients',
                    description: 'Use iteratively reweighted least squares (IRLS). Compute weights wᵢ = min(1, ε/|rᵢ|). Update β = (XᵀWX)⁻¹XᵀWy where W = diag(w).'
                },
                {
                    title: 'RANSAC - Random Sample',
                    description: 'Randomly select minimal subset (for linear regression: 2 points for line, 3 for plane). Fit model to this subset using OLS.'
                },
                {
                    title: 'RANSAC - Find Inliers',
                    description: 'Count how many points are within threshold distance of fitted model. These are "inliers" - points consistent with model.'
                },
                {
                    title: 'RANSAC - Repeat & Select Best',
                    description: 'Repeat sampling N times. Keep model with most inliers. This probabilistically finds the model that explains most data.'
                },
                {
                    title: 'RANSAC - Refit',
                    description: 'Once best inlier set found, refit model using only inliers with OLS. This gives final robust model estimate.'
                },
                {
                    title: 'Theil-Sen - Compute All Slopes',
                    description: 'For each pair of points (xᵢ,yᵢ) and (xⱼ,yⱼ), compute slope mᵢⱼ = (yⱼ-yᵢ)/(xⱼ-xᵢ). This gives C(n,2) slopes.'
                },
                {
                    title: 'Theil-Sen - Median Slope',
                    description: 'The robust slope estimate is the median of all pairwise slopes. Median is inherently resistant to outliers.'
                },
                {
                    title: 'Theil-Sen - Compute Intercept',
                    description: 'Calculate intercept using median of residuals: b = median(yᵢ - m×xᵢ). This ensures overall robustness.'
                }
            ]
        }
    },

    math: {
        objectiveFunction: {
            title: 'Robust Loss Functions',
            formula: 'Multiple formulations depending on method',
            parameters: [
                'OLS Loss: L(r) = r² (sensitive to outliers)',
                'Huber Loss: L(r) = r²/2 if |r|≤ε, else ε|r| - ε²/2',
                'Tukey Bisquare: L(r) = r²(1-(r/c)²)³ if |r|≤c, else 0',
                'ε (epsilon): threshold parameter for Huber',
                'c: tuning constant for other robust losses',
                'r: residual = y - ŷ'
            ]
        },
        additionalContent: [
            {
                title: 'Huber Loss Function',
                formula: 'L_ε(r) = { r²/2  if |r| ≤ ε,  ε|r| - ε²/2  if |r| > ε }',
                description: 'Quadratic near zero (like OLS), linear for large residuals (reduces outlier influence). ε controls transition point.'
            },
            {
                title: 'Huber Objective',
                formula: 'minimize: Σᵢ L_ε(yᵢ - xᵢᵀβ)',
                description: 'Sum of Huber losses over all data points. Solved via iteratively reweighted least squares (IRLS).'
            },
            {
                title: 'RANSAC Inlier Criterion',
                formula: '|yᵢ - xᵢᵀβ| < threshold',
                description: 'Point is inlier if residual below threshold. Threshold typically 2-3× median absolute deviation.'
            },
            {
                title: 'Theil-Sen Estimator',
                formula: 'β = median({(yⱼ-yᵢ)/(xⱼ-xᵢ) : i < j})',
                description: 'Slope is median of all pairwise slopes. Can handle up to 29.3% outliers (breakdown point = 29.3%).'
            },
            {
                title: 'Weight Function (IRLS)',
                formula: 'wᵢ = min(1, ε/|rᵢ|)',
                description: 'Weights for iteratively reweighted least squares. Points with large residuals get smaller weights.'
            }
        ],
        visualization: {
            title: 'Loss Functions Comparison',
            items: [
                {
                    title: 'OLS (L2)',
                    color: '#ef4444',
                    description: 'Squared loss: grows quadratically. Even one outlier far from line dramatically increases total loss, pulling fitted line toward outlier.'
                },
                {
                    title: 'Huber',
                    color: '#f59e0b',
                    description: 'Hybrid: quadratic near origin (efficient for inliers), linear for large residuals (resistant to outliers). Best of both worlds.'
                },
                {
                    title: 'L1 (LAD)',
                    color: '#3b82f6',
                    description: 'Absolute loss: grows linearly. More robust than L2 but can be unstable. Equivalent to median regression.'
                }
            ],
            insight: 'The shape of the loss function determines outlier sensitivity. Quadratic losses (OLS) give outliers huge influence. Linear losses (L1) or capped losses (Huber) limit outlier influence. Huber provides 95% efficiency of OLS on clean data while being robust to outliers.'
        }
    },

    code: {
        examples: [
            {
                title: 'Python (scikit-learn) - Multiple Robust Methods',
                code: `from sklearn.linear_model import HuberRegressor, RANSACRegressor, TheilSenRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Prepare data with outliers
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Add some artificial outliers to demonstrate
n_outliers = int(0.1 * len(y_train))
outlier_idx = np.random.choice(len(y_train), n_outliers, replace=False)
y_train_noisy = y_train.copy()
y_train_noisy[outlier_idx] += np.random.randn(n_outliers) * 50  # Large noise

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Method 1: Huber Regression
huber = HuberRegressor(
    epsilon=1.35,     # Default: 1.35 (95% efficiency)
    max_iter=1000,
    alpha=0.0001,     # L2 regularization
    tol=1e-5
)
huber.fit(X_train_scaled, y_train_noisy)
y_pred_huber = huber.predict(X_test_scaled)

print(f"Huber - Outliers: {np.sum(huber.outliers_)}")
print(f"Huber - R²: {r2_score(y_test, y_pred_huber):.4f}")

# Method 2: RANSAC
ransac = RANSACRegressor(
    min_samples=50,           # Minimum samples for initial fit
    residual_threshold=None,  # Auto: MAD of residuals
    max_trials=100,           # Number of iterations
    random_state=42
)
ransac.fit(X_train_scaled, y_train_noisy)
y_pred_ransac = ransac.predict(X_test_scaled)

# Get inlier mask
inlier_mask = ransac.inlier_mask_
outlier_mask = ~inlier_mask

print(f"\\nRANSAC - Inliers: {np.sum(inlier_mask)}/{len(inlier_mask)}")
print(f"RANSAC - R²: {r2_score(y_test, y_pred_ransac):.4f}")

# Method 3: Theil-Sen Estimator
theilsen = TheilSenRegressor(
    max_subpopulation=1e4,  # Max combinations to try
    n_subsamples=None,      # Auto
    random_state=42
)
theilsen.fit(X_train_scaled, y_train_noisy)
y_pred_theilsen = theilsen.predict(X_test_scaled)

print(f"\\nTheil-Sen - R²: {r2_score(y_test, y_pred_theilsen):.4f}")

# Compare with OLS (for reference)
from sklearn.linear_model import LinearRegression
ols = LinearRegression()
ols.fit(X_train_scaled, y_train_noisy)
y_pred_ols = ols.predict(X_test_scaled)

print(f"\\nOLS (with outliers) - R²: {r2_score(y_test, y_pred_ols):.4f}")

# Visualization
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
plt.scatter(X_train_scaled[:, 0], y_train_noisy, 
           c=ransac.inlier_mask_, cmap='viridis', 
           alpha=0.6, label='Training data')
plt.scatter(X_test_scaled[:, 0], y_test, 
           c='red', alpha=0.3, label='Test data')
plt.plot(X_test_scaled[:, 0], y_pred_ransac, 
        'g-', label='RANSAC', linewidth=2)
plt.plot(X_test_scaled[:, 0], y_pred_huber, 
        'b--', label='Huber', linewidth=2)
plt.plot(X_test_scaled[:, 0], y_pred_ols, 
        'r:', label='OLS', linewidth=2)
plt.legend()
plt.title('Robust Regression Comparison')
plt.show()`
            },
            {
                title: 'R (MASS package)',
                code: `library(MASS)  # For rlm (robust linear model)
library(robustbase)  # More robust methods

# Fit Huber regression
huber_model <- rlm(
  y ~ .,
  data = train_data,
  psi = psi.huber,  # Huber's psi function
  k = 1.345         # Tuning constant
)

summary(huber_model)

# Get weights (lower for outliers)
weights <- huber_model$w
outliers <- which(weights < 0.5)
print(paste("Detected outliers:", length(outliers)))

# Predictions
predictions_huber <- predict(huber_model, newdata = test_data)

# Bisquare (Tukey) method - more aggressive
bisquare_model <- rlm(
  y ~ .,
  data = train_data,
  psi = psi.bisquare,
  c = 4.685  # Tuning constant
)

# MM-estimator (high breakdown, high efficiency)
mm_model <- lmrob(
  y ~ .,
  data = train_data,
  method = "MM"
)

summary(mm_model)

# Compare models
cat("Huber RMSE:", sqrt(mean((test_data$y - predictions_huber)^2)), "\\n")

# Diagnostic plots
par(mfrow = c(2, 2))
plot(huber_model)

# Weight plot to identify outliers
plot(huber_model$w, main = "Robustness Weights",
     ylab = "Weight", xlab = "Observation")
abline(h = 0.5, col = "red", lty = 2)`
            },
            {
                title: 'Python (From Scratch) - Huber Regression',
                code: `import numpy as np

class HuberRegressionScratch:
    def __init__(self, epsilon=1.35, max_iter=100, tol=1e-5):
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None
        self.intercept_ = 0
        
    def huber_loss(self, residuals):
        """Compute Huber loss"""
        abs_res = np.abs(residuals)
        quadratic = (abs_res <= self.epsilon)
        linear = ~quadratic
        
        loss = np.sum(
            quadratic * 0.5 * residuals**2 +
            linear * (self.epsilon * abs_res - 0.5 * self.epsilon**2)
        )
        return loss
    
    def compute_weights(self, residuals):
        """Compute IRLS weights"""
        abs_res = np.abs(residuals)
        weights = np.ones_like(abs_res)
        
        # For large residuals, downweight
        large_res = abs_res > self.epsilon
        weights[large_res] = self.epsilon / abs_res[large_res]
        
        return weights
    
    def fit(self, X, y):
        """Fit using Iteratively Reweighted Least Squares"""
        n_samples, n_features = X.shape
        
        # Add intercept column
        X_with_intercept = np.column_stack([np.ones(n_samples), X])
        
        # Initialize with OLS
        beta = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
        
        # IRLS iterations
        for iteration in range(self.max_iter):
            # Compute residuals
            predictions = X_with_intercept @ beta
            residuals = y - predictions
            
            # Compute weights
            weights = self.compute_weights(residuals)
            
            # Weighted least squares update
            W = np.diag(weights)
            XtWX = X_with_intercept.T @ W @ X_with_intercept
            XtWy = X_with_intercept.T @ W @ y
            
            beta_new = np.linalg.solve(XtWX, XtWy)
            
            # Check convergence
            if np.sum(np.abs(beta_new - beta)) < self.tol:
                print(f"Converged after {iteration + 1} iterations")
                break
                
            beta = beta_new
        
        # Store results
        self.intercept_ = beta[0]
        self.coef_ = beta[1:]
        
        # Mark outliers
        final_residuals = y - X_with_intercept @ beta
        self.outliers_ = np.abs(final_residuals) > self.epsilon
        
        return self
    
    def predict(self, X):
        return X @ self.coef_ + self.intercept_

# Example usage
model = HuberRegressionScratch(epsilon=1.35, max_iter=100)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"Coefficients: {model.coef_}")
print(f"Outliers detected: {np.sum(model.outliers_)}")`
            }
        ]
    },

    preprocessing: {
        critical: {
            title: 'Critical: Don\'t Remove Outliers First!',
            description: 'The whole point of robust regression is to handle outliers automatically. DON\'T manually remove outliers before fitting - let the robust method handle them.',
            code: `# ✗ WRONG: Don't do this\n# Remove outliers manually, then use robust regression\nX_clean = remove_outliers(X_train)  # Defeats the purpose!\nmodel.fit(X_clean, y_clean)\n\n# ✓ CORRECT: Let robust method handle outliers\nmodel.fit(X_train, y_train)  # Includes outliers`,
            why: 'Robust methods are designed to automatically handle outliers. If you remove them first, you\'re not gaining any benefit from using robust regression over OLS.'
        },
        stepsTitle: 'Recommended Preprocessing Pipeline',
        steps: [
            {
                title: 'Handle Missing Values',
                description: 'Robust methods still need complete data.',
                code: `from sklearn.impute import SimpleImputer\n\n# Use median (more robust than mean)\nimputer = SimpleImputer(strategy='median')\nX_imputed = imputer.fit_transform(X)`
            },
            {
                title: 'Feature Scaling (Optional)',
                description: 'Some robust methods benefit from scaling, though they\'re less sensitive than OLS.',
                code: `from sklearn.preprocessing import StandardScaler\n\nscaler = StandardScaler()\nX_scaled = scaler.fit_transform(X)`
            },
            {
                title: 'Encode Categoricals',
                description: 'Standard encoding applies.',
                code: `from sklearn.preprocessing import OneHotEncoder\n\nencoder = OneHotEncoder(drop='first', sparse=False)\nX_encoded = encoder.fit_transform(X_categorical)`
            },
            {
                title: 'Keep Outliers!',
                description: 'This is the key difference - don\'t remove outliers.',
                code: `# Don't do outlier removal like:\n# X = X[abs(X - X.mean()) < 3*X.std()]  # NO!\n\n# Instead, just use the robust method\nmodel.fit(X, y)  # Let it handle outliers`
            }
        ],
        mistakes: {
            title: 'Common Preprocessing Mistakes',
            items: [
                'Removing outliers before fitting - defeats the purpose!',
                'Using mean imputation instead of median',
                'Assuming all methods need scaling (RANSAC less sensitive)',
                'Not checking if outliers are informative vs. errors',
                'Ignoring multicollinearity (still an issue)',
                'Applying robust methods when data is already clean'
            ]
        }
    },

    tips: {
        hyperparameterTuning: {
            title: 'Hyperparameter Tuning Strategy',
            sections: [
                {
                    title: 'Huber Epsilon (ε)',
                    points: [
                        'ε = 1.35: Standard choice (95% efficiency on clean data)',
                        'Smaller ε (e.g., 1.0): More aggressive outlier downweighting',
                        'Larger ε (e.g., 2.0): Less aggressive, closer to OLS',
                        'Rule of thumb: ε ≈ 1.345 × MAD (median absolute deviation)',
                        'Cross-validate to find optimal ε for your data'
                    ]
                },
                {
                    title: 'RANSAC Parameters',
                    points: [
                        'min_samples: Usually 50-100 for stability',
                        'residual_threshold: Auto (MAD-based) works well',
                        'max_trials: 100-1000 depending on outlier fraction',
                        'Higher outlier % → need more trials',
                        'Stop early if good model found (stop_probability)'
                    ]
                },
                {
                    title: 'Method Selection',
                    points: [
                        'Huber: Best for moderate outliers (5-10%)',
                        'RANSAC: Best for many outliers (>30%)',
                        'Theil-Sen: Best for small datasets, up to 29% outliers',
                        'LAD (L1): Simple alternative, median regression',
                        'Bisquare: More aggressive than Huber'
                    ]
                }
            ]
        },
        bestPractices: {
            dos: [
                'Always visualize residuals to identify outliers',
                'Check robustness weights to see which points downweighted',
                'Compare robust vs OLS to understand outlier impact',
                'Use cross-validation for parameter tuning',
                'Investigate detected outliers - are they errors or real?',
                'Consider multiple robust methods and compare',
                'Use median instead of mean for summary statistics'
            ],
            donts: [
                'Don\'t remove outliers manually before fitting',
                'Don\'t use robust methods on clean data (less efficient)',
                'Don\'t ignore detected outliers - investigate them',
                'Don\'t assume all outliers are errors - some may be real',
                'Don\'t use default parameters without understanding them',
                'Don\'t forget to check model assumptions',
                'Don\'t use RANSAC with too few iterations'
            ]
        },
        advancedTechniques: [
            {
                title: 'Identifying Outliers Post-Fitting',
                description: 'After fitting, examine which points were downweighted',
                code: `# For Huber
outlier_mask = huber.outliers_
print(f"Outliers: {np.sum(outlier_mask)}")
print(f"Outlier indices: {np.where(outlier_mask)[0]}")

# For RANSAC
inliers = ransac.inlier_mask_
outliers = ~inliers
print(f"Inliers: {np.sum(inliers)}, Outliers: {np.sum(outliers)}")

# Visualize
plt.scatter(X_train[inliers], y_train[inliers], c='b', label='Inliers')
plt.scatter(X_train[outliers], y_train[outliers], c='r', label='Outliers')
plt.legend()`
            },
            {
                title: 'Robust vs OLS Comparison',
                description: 'Compare robust and OLS to quantify outlier impact',
                code: `from sklearn.linear_model import LinearRegression

# Fit both
ols = LinearRegression().fit(X_train, y_train)
robust = HuberRegressor().fit(X_train, y_train)

# Compare coefficients
coef_diff = np.abs(ols.coef_ - robust.coef_)
print(f"Coefficient differences: {coef_diff}")
print(f"Max difference: {np.max(coef_diff)}")

# Large differences indicate outlier impact
if np.max(coef_diff) > 0.5:
    print("Outliers significantly affecting OLS!")`
            },
            {
                title: 'Adaptive Thresholding',
                description: 'Automatically set threshold based on data statistics',
                code: `from scipy.stats import median_abs_deviation

# Compute MAD-based threshold
mad = median_abs_deviation(y_train)
threshold = 2.5 * mad  # 2.5-3 MAD is common

# Use in RANSAC
ransac = RANSACRegressor(
    residual_threshold=threshold,
    random_state=42
)
ransac.fit(X_train, y_train)`
            }
        ],
        performance: {
            title: 'Performance Optimization',
            tips: [
                'For large datasets: Use Huber (fastest robust method)',
                'RANSAC: Reduce max_trials if speed critical',
                'Theil-Sen: Only for small datasets (<1000 points)',
                'Consider subsampling for initial outlier detection',
                'Use warm_start in iterative methods',
                'Parallel RANSAC: Run multiple trials in parallel',
                'Early stopping: Monitor convergence and stop early'
            ]
        },
        debugging: {
            title: 'Debugging Common Issues',
            issues: [
                {
                    problem: 'Problem: RANSAC not finding inliers',
                    solution: 'Increase max_trials, adjust residual_threshold, or check if data actually has inlier structure'
                },
                {
                    problem: 'Problem: Huber similar to OLS',
                    solution: 'Either no outliers present (good!), or epsilon too large. Decrease epsilon or check data.'
                },
                {
                    problem: 'Problem: All points marked as outliers',
                    solution: 'Threshold too strict. Increase epsilon (Huber) or residual_threshold (RANSAC).'
                },
                {
                    problem: 'Problem: Theil-Sen very slow',
                    solution: 'Dataset too large. Use max_subpopulation parameter or switch to Huber/RANSAC.'
                },
                {
                    problem: 'Problem: Model not converging',
                    solution: 'Increase max_iter, check for perfect multicollinearity, scale features'
                }
            ]
        }
    }
};
