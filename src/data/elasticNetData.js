// Data for Elastic Net algorithm
export const elasticNetData = {
  name: 'Elastic Net Regression',
  category: 'Supervised Learning',
  description: 'Linear regression with L1 and L2 regularization',
  badges: [
    { label: 'Regression', color: 'blue' },
    { label: 'Linear Model', color: 'green' },
    { label: 'Regularized', color: 'purple' },
    { label: 'Feature Selection', color: 'yellow' }
  ],
  
  overview: {
    whatIs: {
      title: 'What is Elastic Net?',
      description: 'Elastic Net is a regularized regression method that combines the penalties of Ridge (L2) and Lasso (L1) regression. It\'s particularly useful when dealing with datasets that have multiple correlated features or when you want both feature selection and coefficient shrinkage.',
      highlight: 'Think of it as the "best of both worlds" - it can eliminate irrelevant features like Lasso while handling correlated features better than Lasso alone.'
    },
    
    whenToUse: {
      title: 'When to Use Elastic Net',
      perfectFor: [
        'Datasets with highly correlated features',
        'When you have more features than samples (p > n)',
        'Need automatic feature selection',
        'Want to prevent overfitting',
        'Linear relationships in data',
        'When both Lasso and Ridge seem appropriate'
      ],
      avoidWhen: [
        'Non-linear relationships dominate',
        'Very small datasets (<100 samples)',
        'Need to capture complex interactions',
        'Features are already independent',
        'Maximum interpretability is critical'
      ]
    },
    
    useCases: {
      title: 'Real-World Use Cases',
      cases: [
        {
          icon: '🏠',
          title: 'Real Estate Price Prediction',
          description: 'Predict house prices with correlated features like square footage, number of rooms, and lot size. Elastic Net handles multicollinearity while selecting important features.'
        },
        {
          icon: '🧬',
          title: 'Genomics & Bioinformatics',
          description: 'Identify genes associated with diseases from highly correlated gene expression data. Handles thousands of features with feature selection capability.'
        },
        {
          icon: '📈',
          title: 'Financial Modeling',
          description: 'Predict stock returns or credit risk with correlated economic indicators. Reduces overfitting common in financial time series.'
        },
        {
          icon: '🎯',
          title: 'Marketing Response Prediction',
          description: 'Predict customer response to campaigns with correlated demographic and behavioral features. Identifies key drivers while handling multicollinearity.'
        }
      ]
    },
    
    prosAndCons: {
      title: 'Advantages & Limitations',
      pros: [
        'Handles Multicollinearity: Works well when features are highly correlated',
        'Feature Selection: Automatically eliminates irrelevant features (sets coefficients to zero)',
        'Prevents Overfitting: Regularization reduces model complexity',
        'Flexible: Balances between Ridge and Lasso with α parameter',
        'Works with p > n: Handles more features than samples'
      ],
      cons: [
        'Linear Assumptions: Cannot capture non-linear relationships',
        'Hyperparameter Tuning: Requires tuning α and l1_ratio parameters',
        'Computational Cost: Slower than simple linear regression',
        'Feature Scaling Required: Sensitive to feature scales',
        'Less Interpretable: Coefficients harder to interpret than OLS'
      ]
    },
    
    stepByStep: {
      title: 'Step-by-Step Algorithm',
      steps: [
        {
          title: 'Initialize',
          description: 'Start with initial coefficient values (usually zeros) and set hyperparameters α (regularization strength) and l1_ratio (balance between L1 and L2)'
        },
        {
          title: 'Compute Predictions',
          description: 'Calculate predicted values: ŷ = Xβ where X is feature matrix and β is coefficient vector'
        },
        {
          title: 'Calculate Loss',
          description: 'Compute loss function: L = MSE + α × l1_ratio × ||β||₁ + α × (1-l1_ratio)/2 × ||β||₂²'
        },
        {
          title: 'Compute Gradient',
          description: 'Calculate gradient of loss with respect to coefficients including both L1 and L2 penalty terms'
        },
        {
          title: 'Update Coefficients',
          description: 'Update β using coordinate descent algorithm, which efficiently handles the L1 penalty term'
        },
        {
          title: 'Apply Soft-Thresholding',
          description: 'For each coefficient, apply soft-thresholding operator to enforce sparsity (set small coefficients to zero)'
        },
        {
          title: 'Check Convergence',
          description: 'Check if change in coefficients or loss is below threshold. If not converged, repeat from step 2'
        },
        {
          title: 'Return Model',
          description: 'Return final coefficients β* and intercept term for making predictions'
        }
      ]
    }
  },
  
  math: {
    objectiveFunction: {
      title: 'Objective Function',
      formula: 'minimize: L(β) = ||y - Xβ||² + α × r × ||β||₁ + α × (1-r)/2 × ||β||₂²',
      parameters: [
        'y: Target variable (n × 1 vector)',
        'X: Feature matrix (n × p matrix)',
        'β: Coefficient vector (p × 1 vector)',
        'α: Regularization strength (α ≥ 0)',
        'r: l1_ratio, mixing parameter (0 ≤ r ≤ 1)',
        '||β||₁: L1 norm = Σ|βⱼ| (Lasso penalty)',
        '||β||₂²: Squared L2 norm = Σβⱼ² (Ridge penalty)'
      ]
    },
    additionalContent: [
      {
        title: 'Gradient (Subgradient)',
        formula: '∂L/∂β = -2X^T(y - Xβ) + α × r × sign(β) + α × (1-r) × β',
        description: 'Note: The L1 term uses subgradient (sign function) since it\'s not differentiable at β = 0'
      }
    ],
    visualization: {
      title: 'Regularization in 2D Feature Space',
      items: [
        {
          title: 'Lasso (L1)',
          color: '#ef4444',
          description: 'Diamond-shaped constraint region. Corners touch axes, leading to sparse solutions (some βⱼ = 0)'
        },
        {
          title: 'Ridge (L2)',
          color: '#3b82f6',
          description: 'Circular constraint region. Smooth boundary, all coefficients shrink but rarely become exactly zero'
        },
        {
          title: 'Elastic Net',
          color: '#8b5cf6',
          description: 'Rounded diamond shape. Combines both: gets sparsity from L1 and grouping effect from L2'
        }
      ],
      insight: 'The optimal solution is where the error contours (ellipses from RSS) first touch the constraint region. Elastic Net\'s shape allows it to select features while handling correlated variables better than Lasso alone.'
    }
  },
  
  code: {
    examples: [
      {
        title: 'Python (scikit-learn)',
        code: `from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Split and scale data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Basic Elastic Net
model = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# With automatic hyperparameter tuning
model_cv = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=5)
model_cv.fit(X_train_scaled, y_train)
print(f"Optimal alpha: {model_cv.alpha_}")`
      },
      {
        title: 'R (glmnet)',
        code: `library(glmnet)

# Cross-validation to find optimal lambda
cv_elastic <- cv.glmnet(X_train, y_train, alpha=0.5, nfolds=5)
best_lambda <- cv_elastic$lambda.min

# Final model
final_model <- glmnet(X_train, y_train, alpha=0.5, lambda=best_lambda)
predictions <- predict(final_model, X_test)`
      }
    ]
  },
  
  preprocessing: {
    critical: {
      title: 'Critical: Feature Scaling',
      description: 'Elastic Net is HIGHLY sensitive to feature scales! Always standardize features before training.',
      code: `from sklearn.preprocessing import StandardScaler\n\nscaler = StandardScaler()\nX_train_scaled = scaler.fit_transform(X_train)\nX_test_scaled = scaler.transform(X_test)`,
      why: 'Features with larger scales will dominate the penalty terms, leading to biased coefficient estimates.'
    },
    stepsTitle: 'Recommended Preprocessing Pipeline',
    steps: [
      {
        title: 'Handle Missing Values',
        description: 'Elastic Net cannot handle missing values directly.',
        code: `from sklearn.impute import SimpleImputer\nimputer = SimpleImputer(strategy='mean')\nX_imputed = imputer.fit_transform(X)`
      },
      {
        title: 'Encode Categorical Variables',
        description: 'Convert categorical features to numerical format.',
        code: `from sklearn.preprocessing import OneHotEncoder\nencoder = OneHotEncoder(drop='first', sparse=False)\nX_encoded = encoder.fit_transform(X_categorical)`
      },
      {
        title: 'Standardization',
        description: 'Transform features to have mean=0 and std=1.',
        code: `from sklearn.preprocessing import StandardScaler\nscaler = StandardScaler()\nX_scaled = scaler.fit_transform(X)`
      }
    ],
    mistakes: {
      title: 'Common Preprocessing Mistakes',
      items: [
        'Forgetting to scale features - leads to biased coefficients',
        'Fitting scaler on test data - causes data leakage',
        'Using different scalers for train and test - inconsistent scaling',
        'Scaling target variable y - not necessary for Elastic Net',
        'Removing all correlated features - Elastic Net handles correlation well'
      ]
    }
  },
  
  tips: {
    hyperparameterTuning: {
      title: 'Hyperparameter Tuning Strategy',
      sections: [
        {
          title: 'Finding Optimal Alpha (α)',
          points: [
            'Start with logarithmically spaced values: [0.001, 0.01, 0.1, 1, 10, 100]',
            'Use ElasticNetCV which automatically tests many alpha values',
            'Smaller α = less regularization (risk overfitting)',
            'Larger α = more regularization (risk underfitting)',
            'Plot cross-validation error vs. alpha to visualize'
          ]
        },
        {
          title: 'Choosing l1_ratio',
          points: [
            'Start with 0.5 (equal L1 and L2 weight)',
            'Try [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]',
            'Higher values (→1) = more Lasso-like (more feature selection)',
            'Lower values (→0) = more Ridge-like (better with correlated features)',
            'Use 0.99 instead of 1.0 to maintain numerical stability'
          ]
        }
      ]
    },
    bestPractices: {
      dos: [
        'Always use cross-validation for hyperparameter selection',
        'Standardize features before fitting',
        'Use ElasticNetCV for automatic alpha selection',
        'Check convergence warnings and increase max_iter if needed',
        'Examine which features have non-zero coefficients',
        'Plot regularization path to understand feature importance'
      ],
      donts: [
        'Forgetting to scale features',
        'Using too large alpha (all coefficients become zero)',
        'Not checking for convergence',
        'Fitting on entire dataset before cross-validation',
        'Comparing coefficients across different l1_ratio values',
        'Expecting perfect feature selection with noisy data'
      ]
    },
    advancedTechniques: [
      {
        title: 'Visualize Regularization Path',
        description: 'See how coefficients change with different alpha values',
        code: `from sklearn.linear_model import enet_path\nalphas, coefs, _ = enet_path(X_train_scaled, y_train, l1_ratio=0.5, n_alphas=100)`
      },
      {
        title: 'Feature Importance Analysis',
        description: 'Rank features by absolute coefficient values',
        code: `feature_importance = pd.DataFrame({'feature': feature_names, 'coefficient': model.coef_})\nfeature_importance['abs_coef'] = abs(feature_importance['coefficient'])\nfeature_importance.sort_values('abs_coef', ascending=False)`
      }
    ],
    performance: {
      title: 'Performance Optimization',
      tips: [
        'Large Datasets: Use SGDRegressor with elastic net penalty for mini-batch training',
        'Many Features: Consider initial feature selection with variance threshold',
        'Sparse Data: Elastic Net works well with sparse matrices - use scipy.sparse format',
        'Parallel Processing: Use n_jobs=-1 in GridSearchCV to utilize all CPU cores',
        'Early Stopping: Set tol parameter appropriately to stop when converged'
      ]
    },
    debugging: {
      title: 'Debugging Common Issues',
      issues: [
        {
          problem: 'Problem: All coefficients are zero',
          solution: 'Alpha is too large. Decrease alpha value or use ElasticNetCV'
        },
        {
          problem: 'Problem: Model not converging',
          solution: 'Increase max_iter or check if features are scaled'
        },
        {
          problem: 'Problem: Poor performance on test set',
          solution: 'Overfitting: increase alpha. Underfitting: decrease alpha or add polynomial features'
        }
      ]
    }
  }
};