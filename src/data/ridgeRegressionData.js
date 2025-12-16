// Data for Ridge Regression (L2) algorithm
export const ridgeRegressionData = {
  name: 'Ridge Regression (L2)',
  category: 'Supervised Learning',
  description: 'Linear regression with L2 regularization to prevent overfitting',
  badges: [
    { label: 'Regression', color: 'blue' },
    { label: 'Regularized', color: 'purple' },
    { label: 'Multicollinearity Handler', color: 'green' },
    { label: 'Shrinkage Method', color: 'orange' }
  ],
  
  overview: {
    whatIs: {
      title: 'What is Ridge Regression?',
      description: 'Ridge Regression (also called Tikhonov regularization or L2 regularization) is a technique that adds a penalty term equal to the square of the magnitude of coefficients to the ordinary least squares objective. Unlike Lasso which can eliminate features, Ridge shrinks all coefficients toward zero but never sets them to exactly zero. This makes it particularly effective when dealing with multicollinearity (correlated features) and when you believe all features contribute to the prediction.',
      highlight: 'Ridge\'s specialty: It keeps all features but shrinks their coefficients, making the model more stable and less sensitive to small changes in the data. Perfect when all features matter!'
    },
    
    whenToUse: {
      title: 'When to Use Ridge Regression',
      perfectFor: [
        'Features are highly correlated (multicollinearity)',
        'All features are believed to be relevant',
        'More features than samples (p > n)',
        'Want to keep all features but reduce overfitting',
        'Model stability is important',
        'Dealing with ill-conditioned design matrices',
        'Need smooth coefficient estimates',
        'Prior knowledge suggests all features contribute'
      ],
      avoidWhen: [
        'Need feature selection (use Lasso instead)',
        'Many features are truly irrelevant',
        'Interpretability requires sparse models',
        'Computational resources very limited',
        'Features are already orthogonal',
        'Want to identify key drivers (use Lasso/Elastic Net)'
      ]
    },
    
    useCases: {
      title: 'Real-World Use Cases',
      cases: [
        {
          icon: '🏠',
          title: 'Real Estate Price Prediction',
          description: 'Predict house prices using correlated features (square footage, rooms, lot size). Ridge handles the correlation well and keeps all features, providing stable predictions across different neighborhoods.'
        },
        {
          icon: '💊',
          title: 'Drug Discovery & Molecular Modeling',
          description: 'Predict drug effectiveness from molecular descriptors. Thousands of correlated chemical properties require Ridge to maintain stability. All properties may contribute to effectiveness in complex ways.'
        },
        {
          icon: '📈',
          title: 'Stock Portfolio Optimization',
          description: 'Model asset returns using economic indicators that are highly correlated. Ridge provides stable coefficient estimates even when indicators move together, avoiding extreme portfolio weights.'
        },
        {
          icon: '🌦️',
          title: 'Weather Forecasting',
          description: 'Predict temperature, rainfall using correlated atmospheric measurements (pressure, humidity, wind). All measurements provide information, but correlations cause instability without regularization.'
        },
        {
          icon: '🔊',
          title: 'Signal Processing & Noise Reduction',
          description: 'Reconstruct signals from noisy measurements. Ridge regression provides stable solutions to ill-posed inverse problems common in signal processing, imaging, and acoustics.'
        },
        {
          icon: '🧬',
          title: 'Genomics with Dense Signals',
          description: 'Predict phenotypes when many genes contribute small effects. Unlike Lasso which would eliminate most genes, Ridge keeps all genes with small coefficients, matching biological reality.'
        }
      ]
    },
    
    prosAndCons: {
      title: 'Advantages & Limitations',
      pros: [
        'Handles Multicollinearity: Works well with highly correlated features',
        'Stable Estimates: Small data changes produce small coefficient changes',
        'Closed-Form Solution: Can be solved analytically (unlike Lasso)',
        'Always Unique: Single optimal solution (no ambiguity)',
        'Computational Efficiency: Fast matrix operations',
        'Works with p > n: Handles more features than samples',
        'Smooth Shrinkage: All coefficients shrink proportionally',
        'No Feature Instability: Always uses all features'
      ],
      cons: [
        'No Feature Selection: Keeps all features (less interpretable)',
        'Model Complexity: Cannot eliminate irrelevant features',
        'Storage Requirements: Must store all coefficients',
        'Interpretation Difficulty: Hard to identify key features',
        'Biased Estimates: Introduces bias to reduce variance',
        'Scaling Sensitive: Requires standardized features',
        'Not Sparse: Cannot achieve sparsity in high dimensions'
      ]
    },
    
    stepByStep: {
      title: 'Step-by-Step Algorithm',
      steps: [
        {
          title: 'Set Up Problem',
          description: 'Define objective: minimize ||y - Xβ||² + α||β||² where α is regularization strength. Choose α through cross-validation or domain knowledge.'
        },
        {
          title: 'Add Ridge Term to Normal Equations',
          description: 'Modify normal equations: (XᵀX + αI)β = Xᵀy where I is identity matrix. The αI term is the ridge penalty added to XᵀX.'
        },
        {
          title: 'Check Matrix Condition',
          description: 'XᵀX + αI is always invertible (positive definite) as long as α > 0. This solves the multicollinearity problem where XᵀX might be singular.'
        },
        {
          title: 'Solve Linear System',
          description: 'Compute β = (XᵀX + αI)⁻¹Xᵀy using Cholesky decomposition or SVD. This is a standard linear algebra operation, fast and stable.'
        },
        {
          title: 'Alternative: Use SVD',
          description: 'For numerical stability, decompose X = UΣVᵀ. Then β = V(Σ² + αI)⁻¹ΣUᵀy. SVD avoids forming XᵀX which can be ill-conditioned.'
        },
        {
          title: 'Compute Intercept',
          description: 'If data is centered, intercept = ȳ (mean of y). If not centered, include intercept in model but do not penalize it.'
        },
        {
          title: 'Make Predictions',
          description: 'For new data x*, compute ŷ* = x*ᵀβ + intercept. Predictions are stable even with correlated features.'
        },
        {
          title: 'Evaluate Regularization',
          description: 'Check ||β||² (sum of squared coefficients). As α increases, ||β||² decreases. Balance between fit and coefficient magnitude.'
        }
      ]
    }
  },
  
  math: {
    objectiveFunction: {
      title: 'Ridge Regression Objective',
      formula: 'minimize: L(β) = ||y - Xβ||² + α||β||²',
      parameters: [
        'y: Target variable (n × 1 vector)',
        'X: Feature matrix (n × p matrix)',
        'β: Coefficient vector (p × 1 vector)',
        'α: Regularization parameter (α ≥ 0)',
        '||β||²: L2 norm squared = βᵀβ = Σⱼβⱼ²',
        'First term: Residual sum of squares (RSS)',
        'Second term: L2 penalty shrinking coefficients',
        'Also written as: RSS + α||β||₂²'
      ]
    },
    additionalContent: [
      {
        title: 'Closed-Form Solution',
        formula: 'β_ridge = (XᵀX + αI)⁻¹Xᵀy',
        description: 'Exact analytical solution. Adding αI makes matrix invertible even when XᵀX is singular (p > n or multicollinearity).'
      },
      {
        title: 'SVD-Based Solution',
        formula: 'β_ridge = VDUᵀy where D = diag(dᵢ/(dᵢ² + α))',
        description: 'Numerically stable computation using Singular Value Decomposition X = UΣVᵀ. dᵢ are singular values of X.'
      },
      {
        title: 'Relationship to OLS',
        formula: 'β_ridge = (XᵀX + αI)⁻¹Xᵀy vs β_OLS = (XᵀX)⁻¹Xᵀy',
        description: 'Ridge adds αI to XᵀX. As α→0, Ridge→OLS. As α→∞, β_ridge→0.'
      },
      {
        title: 'Effective Degrees of Freedom',
        formula: 'df(α) = Σᵢ dᵢ²/(dᵢ² + α) where dᵢ are singular values',
        description: 'Measures model complexity. df(0)=p (OLS), df(∞)=0. Useful for model selection.'
      }
    ],
    visualization: {
      title: 'Ridge Constraint Geometry',
      items: [
        {
          title: 'L2 Ball (Sphere)',
          color: '#3b82f6',
          description: 'L2 constraint ||β||² ≤ t forms a circle in 2D, sphere in higher dimensions. Smooth boundary with no corners.'
        },
        {
          title: 'RSS Contours',
          color: '#ef4444',
          description: 'Residual sum of squares forms elliptical contours centered at OLS solution. Elongated along correlated feature directions.'
        },
        {
          title: 'Ridge Solution',
          color: '#8b5cf6',
          description: 'Optimal solution where RSS contour tangent to L2 sphere. Never touches axes (no sparsity), all coefficients non-zero but shrunk.'
        }
      ],
      insight: 'The circular L2 constraint has no corners, so the solution rarely lands on axes. This means Ridge shrinks all coefficients but never eliminates features. Larger α = smaller sphere = more shrinkage.'
    }
  },
  
  code: {
    examples: [
      {
        title: 'Python (scikit-learn)',
        code: `from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

# Prepare data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Basic Ridge
ridge = Ridge(alpha=1.0, fit_intercept=True, solver='auto')
ridge.fit(X_train_scaled, y_train)
y_pred = ridge.predict(X_test_scaled)

print(f"All {len(ridge.coef_)} coefficients non-zero")
print(f"Coefficient L2 norm: {np.linalg.norm(ridge.coef_):.4f}")

# Automatic alpha selection
ridge_cv = RidgeCV(alphas=np.logspace(-3, 3, 100), cv=5)
ridge_cv.fit(X_train_scaled, y_train)
print(f"Optimal alpha: {ridge_cv.alpha_:.4f}")`
      },
      {
        title: 'R (glmnet)',
        code: `library(glmnet)

# Scale features
X_scaled <- scale(X_train)

# Fit Ridge (alpha = 0)
ridge_model <- glmnet(X_scaled, y_train, alpha = 0, lambda = 1.0)

# Cross-validation
cv_ridge <- cv.glmnet(X_scaled, y_train, alpha = 0, nfolds = 5)
plot(cv_ridge)

best_lambda <- cv_ridge$lambda.min
final_model <- glmnet(X_scaled, y_train, alpha = 0, lambda = best_lambda)
coef(final_model)`
      },
      {
        title: 'Python (From Scratch)',
        code: `import numpy as np

class RidgeRegression:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.coef_ = None
        self.intercept_ = 0
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.intercept_ = np.mean(y)
        y_centered = y - self.intercept_
        
        # Ridge solution: (X^T X + αI)^(-1) X^T y
        XtX = X.T @ X
        identity = np.eye(n_features)
        self.coef_ = np.linalg.solve(XtX + self.alpha * identity, X.T @ y_centered)
        
        return self
    
    def predict(self, X):
        return X @ self.coef_ + self.intercept_

# Usage
ridge = RidgeRegression(alpha=1.0)
ridge.fit(X_train_scaled, y_train)
y_pred = ridge.predict(X_test_scaled)`
      }
    ]
  },
  
  preprocessing: {
    critical: {
      title: 'Important: Feature Scaling',
      description: 'Ridge regression is sensitive to feature scales. Standardization is highly recommended for fair penalization.',
      code: `from sklearn.preprocessing import StandardScaler\n\nscaler = StandardScaler()\nX_train_scaled = scaler.fit_transform(X_train)\nX_test_scaled = scaler.transform(X_test)`,
      why: 'Ridge penalty treats all coefficients equally. Features with larger scales will have artificially small coefficients to minimize the penalty. Standardization ensures fair treatment.'
    },
    stepsTitle: 'Recommended Preprocessing Pipeline',
    steps: [
      {
        title: 'Handle Missing Values',
        description: 'Ridge cannot handle missing values directly.',
        code: `from sklearn.impute import SimpleImputer\n\nimputer = SimpleImputer(strategy='mean')\nX_imputed = imputer.fit_transform(X)`
      },
      {
        title: 'Encode Categorical Variables',
        description: 'Convert categorical features to numerical format.',
        code: `from sklearn.preprocessing import OneHotEncoder\n\nencoder = OneHotEncoder(drop='first', sparse=False)\nX_encoded = encoder.fit_transform(X_categorical)`
      },
      {
        title: 'Standardization',
        description: 'Transform features to mean=0, std=1.',
        code: `from sklearn.preprocessing import StandardScaler\n\nscaler = StandardScaler()\nX_scaled = scaler.fit_transform(X)`
      }
    ],
    completePipeline: {
      title: 'Complete Pipeline',
      code: `from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV

pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('ridge', RidgeCV(alphas=np.logspace(-3, 3, 100), cv=5))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)`
    },
    mistakes: {
      title: 'Common Mistakes',
      items: [
        'Not standardizing features - biases toward larger scale features',
        'Penalizing the intercept - should only penalize slopes',
        'Removing correlated features - Ridge handles them well',
        'Using Ridge when features are orthogonal - no benefit',
        'Forgetting to save scaler for deployment',
        'Not checking if multicollinearity actually exists'
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
            'Small α (0.001): Weak penalty, close to OLS',
            'Large α (1000): Strong penalty, coefficients near zero',
            'Use RidgeCV for automatic selection',
            'Try logarithmically spaced values',
            'Plot CV error vs alpha'
          ]
        },
        {
          title: 'Cross-Validation',
          points: [
            'Use 5-fold or 10-fold CV',
            'Try 100+ alpha values on log scale',
            'Look for elbow in CV error curve',
            'Generalized CV is fast approximation',
            'Monitor effective degrees of freedom'
          ]
        }
      ]
    },
    bestPractices: {
      dos: [
        'Use RidgeCV for automatic alpha selection',
        'Standardize features before fitting',
        'Plot regularization path',
        'Check VIF to confirm multicollinearity',
        'Use SVD solver for stability',
        'Save scaler with model',
        'Compare Ridge vs OLS',
        'Monitor effective degrees of freedom'
      ],
      donts: [
        'Do not use Ridge for feature selection',
        'Do not penalize the intercept',
        'Do not use when features are orthogonal',
        'Do not forget to standardize',
        'Do not compare coefficients across alphas',
        'Do not expect sparse solutions',
        'Do not use when most features irrelevant'
      ]
    },
    advancedTechniques: [
      {
        title: 'Generalized Cross-Validation',
        description: 'Fast approximation to leave-one-out CV',
        code: `from sklearn.linear_model import RidgeCV\n\nridge_gcv = RidgeCV(alphas=np.logspace(-3, 3, 100), cv=None)\nridge_gcv.fit(X_train_scaled, y_train)\nprint(f"GCV optimal alpha: {ridge_gcv.alpha_:.4f}")`
      },
      {
        title: 'Ridge Trace Plot',
        description: 'Visualize coefficient stability',
        code: `import matplotlib.pyplot as plt\n\nalphas = np.logspace(-2, 2, 100)\ncoefs = []\n\nfor alpha in alphas:\n    ridge_temp = Ridge(alpha=alpha)\n    ridge_temp.fit(X_train_scaled, y_train)\n    coefs.append(ridge_temp.coef_)\n\ncoefs = np.array(coefs)\nplt.plot(alphas, coefs)\nplt.xscale('log')\nplt.xlabel('Alpha')\nplt.ylabel('Coefficients')\nplt.title('Ridge Trace')\nplt.show()`
      }
    ],
    performance: {
      title: 'Performance Optimization',
      tips: [
        'Closed-form solution is very fast',
        'Use SVD solver for numerical stability',
        'Cholesky decomposition fastest for well-conditioned problems',
        'Use sparse matrices for high-dimensional data',
        'LSQR solver good for very large datasets',
        'Parallel CV with n_jobs=-1',
        'Pre-compute XᵀX for multiple alphas'
      ]
    },
    debugging: {
      title: 'Debugging Common Issues',
      issues: [
        {
          problem: 'Problem: Ridge performs same as OLS',
          solution: 'Alpha too small. Increase alpha or check if multicollinearity exists. Try alphas in range [0.1, 10].'
        },
        {
          problem: 'Problem: All coefficients near zero',
          solution: 'Alpha too large. Decrease alpha. Check feature scaling. Try smaller values like 0.01.'
        },
        {
          problem: 'Problem: Numerical instability / NaN',
          solution: 'Use SVD solver. Check for extreme values. Ensure features are scaled.'
        },
        {
          problem: 'Problem: Poor test performance',
          solution: 'Alpha too small. Increase alpha through cross-validation. May need more regularization.'
        },
        {
          problem: 'Problem: Cannot decide Ridge vs Lasso',
          solution: 'Use Elastic Net with l1_ratio=0.5. Try both via CV. Ridge if all features matter.'
        },
        {
          problem: 'Problem: Ridge not helping multicollinearity',
          solution: 'Alpha too small. Check VIF values. Use larger alpha. Standardize features first.'
        }
      ]
    }
  }
};