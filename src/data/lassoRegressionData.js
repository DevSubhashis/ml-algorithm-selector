// Data for Lasso Regression (L1) algorithm
export const lassoRegressionData = {
  name: 'Lasso Regression (L1)',
  category: 'Supervised Learning',
  description: 'Linear regression with L1 regularization for automatic feature selection',
  badges: [
    { label: 'Regression', color: 'blue' },
    { label: 'Feature Selection', color: 'green' },
    { label: 'Regularized', color: 'purple' },
    { label: 'Sparse Solutions', color: 'yellow' }
  ],
  
  overview: {
    whatIs: {
      title: 'What is Lasso Regression?',
      description: 'Lasso (Least Absolute Shrinkage and Selection Operator) is a linear regression technique that uses L1 regularization. It adds a penalty term equal to the absolute value of the magnitude of coefficients. The key feature of Lasso is that it can shrink some coefficients to exactly zero, effectively performing automatic feature selection. This makes it particularly useful when you have many features and want to identify the most important ones.',
      highlight: 'Lasso\'s superpower: It automatically eliminates irrelevant features by setting their coefficients to zero. You get both prediction and feature selection in one shot!'
    },
    
    whenToUse: {
      title: 'When to Use Lasso Regression',
      perfectFor: [
        'High-dimensional data with many features',
        'Need automatic feature selection',
        'Suspect many features are irrelevant',
        'Want sparse models (few non-zero coefficients)',
        'Interpretability is important',
        'Features are not highly correlated',
        'Preventing overfitting with many predictors',
        'Building simple, explainable models'
      ],
      avoidWhen: [
        'Features are highly correlated (use Elastic Net)',
        'All features are important (use Ridge)',
        'Need all correlated features in model (Lasso picks one)',
        'More features than samples and features are grouped',
        'Very small datasets (<50 samples)',
        'Non-linear relationships dominate'
      ]
    },
    
    useCases: {
      title: 'Real-World Use Cases',
      cases: [
        {
          icon: '🧬',
          title: 'Genomics & Gene Selection',
          description: 'Identify which genes (out of thousands) are associated with a disease. Lasso automatically selects relevant genes and eliminates noise, making results interpretable for biologists.'
        },
        {
          icon: '📊',
          title: 'Financial Modeling',
          description: 'Select key economic indicators from hundreds of potential predictors for stock returns or risk modeling. Lasso identifies the most influential factors while eliminating redundant ones.'
        },
        {
          icon: '🏥',
          title: 'Medical Diagnosis',
          description: 'Determine which symptoms or biomarkers predict disease from extensive patient data. Lasso creates simple diagnostic rules by selecting only critical indicators.'
        },
        {
          icon: '📱',
          title: 'Marketing Attribution',
          description: 'Identify which marketing channels drive conversions from many possible touchpoints. Lasso reveals which channels truly matter and which are wasting budget.'
        },
        {
          icon: '🏠',
          title: 'Real Estate Pricing',
          description: 'Select important property features from dozens of variables (location, size, amenities). Lasso builds simple pricing models focusing on key value drivers.'
        },
        {
          icon: '🔬',
          title: 'Scientific Feature Discovery',
          description: 'Discover which experimental variables influence outcomes in physics, chemistry, or engineering experiments with many measured parameters.'
        }
      ]
    },
    
    prosAndCons: {
      title: 'Advantages & Limitations',
      pros: [
        'Automatic Feature Selection: Sets irrelevant coefficients to exactly zero',
        'Interpretable Models: Sparse solutions easy to understand and explain',
        'Prevents Overfitting: L1 penalty reduces model complexity',
        'Fast Computation: Efficient algorithms (coordinate descent) available',
        'No Manual Feature Selection: Eliminates need for stepwise selection',
        'Works with p > n: Handles more features than samples',
        'Simple Hyperparameter: Only one parameter (alpha) to tune',
        'Built-in Regularization: Automatically controls model complexity'
      ],
      cons: [
        'Correlated Features Problem: Arbitrarily selects one from correlated group',
        'Instability: Small data changes can change selected features',
        'Limited Selection: Can select at most n features (sample size)',
        'Not Unique: Multiple optimal solutions possible with correlations',
        'Linear Assumptions: Cannot capture non-linear relationships',
        'Bias: Introduces bias to reduce variance (bias-variance tradeoff)',
        'Sensitive to Scaling: Requires standardized features'
      ]
    },
    
    stepByStep: {
      title: 'Step-by-Step Algorithm',
      steps: [
        {
          title: 'Initialize Coefficients',
          description: 'Start with initial coefficient values β = 0 or use OLS estimates. Set regularization parameter α (controls strength of L1 penalty).'
        },
        {
          title: 'Compute Predictions',
          description: 'Calculate predicted values: ŷ = Xβ where X is the feature matrix and β is the coefficient vector.'
        },
        {
          title: 'Calculate Loss with L1 Penalty',
          description: 'Compute objective: L = ||y - Xβ||² + α||β||₁ where ||β||₁ = Σ|βⱼ| is the L1 norm (sum of absolute values).'
        },
        {
          title: 'Coordinate Descent - Select Feature',
          description: 'Pick one coefficient βⱼ to update while keeping others fixed. This makes the problem simpler to solve.'
        },
        {
          title: 'Compute Partial Residual',
          description: 'Calculate residual without feature j: rⱼ = y - Σ(i≠j) Xᵢβᵢ. This isolates the effect of feature j.'
        },
        {
          title: 'Apply Soft-Thresholding',
          description: 'Update coefficient: βⱼ = S(Xⱼᵀrⱼ, α) / ||Xⱼ||² where S(z,γ) = sign(z)max(|z|-γ, 0) is soft-thresholding operator. This can set βⱼ to exactly zero.'
        },
        {
          title: 'Cycle Through All Features',
          description: 'Repeat steps 4-6 for all features j = 1,2,...,p. This completes one full iteration of coordinate descent.'
        },
        {
          title: 'Check Convergence',
          description: 'Check if coefficients changed less than tolerance: ||β_new - β_old|| < tol. If not converged, repeat from step 2. Otherwise, return final β.'
        }
      ]
    }
  },
  
  math: {
    objectiveFunction: {
      title: 'Lasso Objective Function',
      formula: 'minimize: L(β) = (1/2n)||y - Xβ||² + α||β||₁',
      parameters: [
        'y: Target variable (n × 1 vector)',
        'X: Feature matrix (n × p matrix)',
        'β: Coefficient vector (p × 1 vector)',
        'α: Regularization strength (α ≥ 0)',
        '||β||₁: L1 norm = Σⱼ|βⱼ| (sum of absolute values)',
        'n: Number of samples',
        'First term: Residual sum of squares (RSS)',
        'Second term: L1 penalty promoting sparsity'
      ]
    },
    additionalContent: [
      {
        title: 'L1 Norm (Manhattan Distance)',
        formula: '||β||₁ = |β₁| + |β₂| + ... + |βₚ| = Σⱼ|βⱼ|',
        description: 'Sum of absolute values of coefficients. Creates diamond-shaped constraint region that has corners on axes, leading to sparse solutions.'
      },
      {
        title: 'Soft-Thresholding Operator',
        formula: 'S(z, γ) = sign(z) × max(|z| - γ, 0)',
        description: 'Key operation in Lasso. Shrinks z toward zero by γ, and sets to zero if |z| ≤ γ. This creates sparsity.'
      },
      {
        title: 'Coordinate Descent Update',
        formula: 'βⱼ = S(Xⱼᵀrⱼ, nα) / ||Xⱼ||²',
        description: 'Update rule for coefficient j. rⱼ is partial residual. Soft-threshold applied with penalty nα.'
      },
      {
        title: 'Subgradient',
        formula: '∂||β||₁/∂βⱼ = sign(βⱼ) if βⱼ≠0, ∈[-1,1] if βⱼ=0',
        description: 'L1 norm is not differentiable at zero. We use subgradient instead. This allows some coefficients to be exactly zero.'
      }
    ],
    visualization: {
      title: 'Lasso Constraint Region',
      items: [
        {
          title: 'L1 Ball (Diamond)',
          color: '#ef4444',
          description: 'L1 constraint ||β||₁ ≤ t forms a diamond shape in 2D. Has sharp corners on coordinate axes.'
        },
        {
          title: 'RSS Contours',
          color: '#3b82f6',
          description: 'Residual sum of squares forms elliptical contours centered at OLS solution.'
        },
        {
          title: 'Lasso Solution',
          color: '#8b5cf6',
          description: 'Optimal solution where RSS contour first touches L1 ball. Often touches at a corner, making some βⱼ = 0.'
        }
      ],
      insight: 'The diamond shape of L1 constraint has corners on coordinate axes. RSS contours often hit these corners first, resulting in coefficients being exactly zero. This is why Lasso does feature selection while Ridge only shrinks.'
    }
  },
  
  code: {
    examples: [
      {
        title: 'Python (scikit-learn)',
        code: `from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

# Prepare data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# CRITICAL: Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Basic Lasso
lasso = Lasso(alpha=0.1, max_iter=10000)
lasso.fit(X_train_scaled, y_train)
y_pred = lasso.predict(X_test_scaled)

# Feature selection results
n_features = np.sum(lasso.coef_ != 0)
print(f"Features selected: {n_features}/{len(lasso.coef_)}")

# Automatic alpha selection
lasso_cv = LassoCV(cv=5, max_iter=10000, n_jobs=-1)
lasso_cv.fit(X_train_scaled, y_train)
print(f"Optimal alpha: {lasso_cv.alpha_:.6f}")`
      },
      {
        title: 'R (glmnet)',
        code: `library(glmnet)

# Scale features
X_scaled <- scale(X_train)

# Fit Lasso (alpha = 1)
lasso_model <- glmnet(X_scaled, y_train, alpha = 1, lambda = 0.1)

# Cross-validation for optimal lambda
cv_lasso <- cv.glmnet(X_scaled, y_train, alpha = 1, nfolds = 5)
plot(cv_lasso)

best_lambda <- cv_lasso$lambda.min
final_model <- glmnet(X_scaled, y_train, alpha = 1, lambda = best_lambda)

# Coefficients (many will be zero)
coef(final_model)`
      },
      {
        title: 'Python (From Scratch)',
        code: `import numpy as np

class LassoRegression:
    def __init__(self, alpha=1.0, max_iter=1000, tol=1e-4):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None
        self.intercept_ = 0
        
    def soft_threshold(self, x, threshold):
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.intercept_ = np.mean(y)
        y_centered = y - self.intercept_
        self.coef_ = np.zeros(n_features)
        
        for iteration in range(self.max_iter):
            coef_old = self.coef_.copy()
            
            for j in range(n_features):
                residual = y_centered - X @ self.coef_ + X[:, j] * self.coef_[j]
                rho = X[:, j] @ residual
                self.coef_[j] = self.soft_threshold(rho / n_samples, self.alpha)
            
            if np.sum(np.abs(self.coef_ - coef_old)) < self.tol:
                break
        
        return self
    
    def predict(self, X):
        return X @ self.coef_ + self.intercept_

# Usage
lasso = LassoRegression(alpha=0.1)
lasso.fit(X_train_scaled, y_train)
y_pred = lasso.predict(X_test_scaled)`
      }
    ]
  },
  
  preprocessing: {
    critical: {
      title: 'Critical: Feature Standardization',
      description: 'Lasso is EXTREMELY sensitive to feature scales! Always standardize features before fitting.',
      code: `from sklearn.preprocessing import StandardScaler\n\nscaler = StandardScaler()\nX_train_scaled = scaler.fit_transform(X_train)\nX_test_scaled = scaler.transform(X_test)`,
      why: 'Lasso penalty treats all coefficients equally regardless of feature scale. Without standardization, features with larger scales will be incorrectly penalized and eliminated.'
    },
    stepsTitle: 'Recommended Preprocessing Pipeline',
    steps: [
      {
        title: 'Handle Missing Values',
        description: 'Lasso cannot handle missing values.',
        code: `from sklearn.impute import SimpleImputer\n\nimputer = SimpleImputer(strategy='mean')\nX_imputed = imputer.fit_transform(X)`
      },
      {
        title: 'Encode Categorical Variables',
        description: 'Convert categoricals to numerical.',
        code: `from sklearn.preprocessing import OneHotEncoder\n\nencoder = OneHotEncoder(drop='first', sparse=False)\nX_encoded = encoder.fit_transform(X_categorical)`
      },
      {
        title: 'Standardization (CRITICAL)',
        description: 'Transform to mean=0, std=1.',
        code: `from sklearn.preprocessing import StandardScaler\n\nscaler = StandardScaler()\nX_scaled = scaler.fit_transform(X)`
      }
    ],
    completePipeline: {
      title: 'Complete Pipeline',
      code: `from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV

pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('lasso', LassoCV(cv=5, max_iter=10000))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)`
    },
    mistakes: {
      title: 'Common Mistakes',
      items: [
        'NOT standardizing features - breaks feature selection',
        'Standardizing test data independently - causes data leakage',
        'Not checking for highly correlated features',
        'Using Lasso when all features are important',
        'Forgetting to save scaler for deployment'
      ]
    }
  },
  
  tips: {
    hyperparameterTuning: {
      title: 'Hyperparameter Tuning',
      sections: [
        {
          title: 'Choosing Alpha',
          points: [
            'Small α (0.001): Weak penalty, more features',
            'Large α (10): Strong penalty, few features',
            'Use LassoCV for automatic selection',
            'Try logarithmically spaced values',
            'Plot CV error vs alpha'
          ]
        },
        {
          title: 'Cross-Validation',
          points: [
            'Use 5-fold or 10-fold CV',
            'Try 100+ alpha values',
            'Monitor number of selected features',
            'One-standard-error rule for simpler models'
          ]
        }
      ]
    },
    bestPractices: {
      dos: [
        'ALWAYS standardize features',
        'Use LassoCV for automatic alpha selection',
        'Plot regularization path',
        'Check feature stability across CV folds',
        'Save scaler with model',
        'Visualize selected features'
      ],
      donts: [
        'Never skip standardization',
        'Do not use with highly correlated features',
        'Do not expect stable selection with small samples',
        'Do not use if all features are important',
        'Do not forget categorical encoding increases features'
      ]
    },
    advancedTechniques: [
      {
        title: 'Stability Selection',
        description: 'Run Lasso on bootstrap samples to find stable features',
        code: `from sklearn.utils import resample\n\nstability = np.zeros(n_features)\nfor i in range(100):\n    X_boot, y_boot = resample(X_train, y_train)\n    lasso.fit(X_boot, y_boot)\n    stability += (lasso.coef_ != 0)\n\nstable_features = np.where(stability > 50)[0]`
      },
      {
        title: 'Regularization Path',
        description: 'Visualize coefficient paths',
        code: `from sklearn.linear_model import lasso_path\n\nalphas, coefs, _ = lasso_path(X_train, y_train, n_alphas=100)\nplt.plot(alphas, coefs.T)\nplt.xscale('log')\nplt.xlabel('Alpha')\nplt.ylabel('Coefficients')`
      }
    ],
    performance: {
      title: 'Performance Optimization',
      tips: [
        'Use coordinate descent (scikit-learn default)',
        'Warm start for multiple alphas',
        'Use sparse matrices for high-dimensional data',
        'Set appropriate tol for early stopping',
        'Parallel CV with n_jobs=-1'
      ]
    },
    debugging: {
      title: 'Debugging Common Issues',
      issues: [
        {
          problem: 'Problem: All coefficients are zero',
          solution: 'Alpha too large. Decrease alpha or use LassoCV.'
        },
        {
          problem: 'Problem: Too many features selected',
          solution: 'Alpha too small. Increase alpha for more sparsity.'
        },
        {
          problem: 'Problem: Important features eliminated',
          solution: 'Check feature scaling. Verify features are standardized.'
        },
        {
          problem: 'Problem: Unstable feature selection',
          solution: 'Use Elastic Net or stability selection. May indicate correlations.'
        },
        {
          problem: 'Problem: Model not converging',
          solution: 'Increase max_iter to 10000. Check feature scaling.'
        },
        {
          problem: 'Problem: Only one correlated feature selected',
          solution: 'Expected for Lasso. Use Elastic Net to keep correlated features.'
        }
      ]
    }
  }
};