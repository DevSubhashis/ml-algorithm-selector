// Data for Bayesian Linear Regression algorithm
export const bayesianLinearData = {
    name: 'Bayesian Linear Regression',
    category: 'Supervised Learning',
    description: 'Probabilistic approach to linear regression with uncertainty quantification',
    badges: [
        { label: 'Regression', color: 'blue' },
        { label: 'Probabilistic', color: 'purple' },
        { label: 'Uncertainty Estimation', color: 'green' },
        { label: 'Bayesian', color: 'indigo' }
    ],

    overview: {
        whatIs: {
            title: 'What is Bayesian Linear Regression?',
            description: 'Bayesian Linear Regression is a probabilistic approach to linear regression that treats model parameters as random variables with probability distributions rather than fixed values. Unlike traditional linear regression which gives point estimates, Bayesian regression provides full posterior distributions over parameters, enabling uncertainty quantification in predictions.',
            highlight: 'The key advantage: You get not just a prediction, but also confidence intervals showing how certain the model is about each prediction. Perfect for decision-making under uncertainty!'
        },

        whenToUse: {
            title: 'When to Use Bayesian Linear Regression',
            perfectFor: [
                'Need uncertainty estimates for predictions',
                'Small datasets where confidence is important',
                'Sequential/online learning scenarios',
                'Incorporating prior knowledge about parameters',
                'Making decisions with risk assessment',
                'Regularization through prior distributions',
                'When interpretability with uncertainty is critical'
            ],
            avoidWhen: [
                'Very large datasets (computationally expensive)',
                'Only point predictions needed (use OLS instead)',
                'No prior knowledge available for informative priors',
                'Real-time predictions required (slower inference)',
                'Non-linear relationships dominate'
            ]
        },

        useCases: {
            title: 'Real-World Use Cases',
            cases: [
                {
                    icon: '⚕️',
                    title: 'Medical Diagnosis & Treatment Planning',
                    description: 'Predict patient outcomes with confidence intervals. Doctors need to know not just the prediction but also how certain the model is. Bayesian regression provides credible intervals for risk assessment in treatment decisions.'
                },
                {
                    icon: '💰',
                    title: 'Financial Risk Assessment',
                    description: 'Portfolio optimization and risk management require understanding prediction uncertainty. Bayesian regression quantifies uncertainty in expected returns, helping with better risk-adjusted decisions.'
                },
                {
                    icon: '🏭',
                    title: 'Quality Control & Manufacturing',
                    description: 'Predict product quality with uncertainty bounds. When predictions indicate high uncertainty, additional quality checks can be triggered. Incorporates expert knowledge through priors.'
                },
                {
                    icon: '🔬',
                    title: 'Scientific Experiments',
                    description: 'Analyze experimental data with small sample sizes. Bayesian methods naturally handle uncertainty in low-data regimes and allow incorporation of domain knowledge from literature.'
                },
                {
                    icon: '🌡️',
                    title: 'Environmental Monitoring',
                    description: 'Predict temperature, pollution levels with confidence bands. Critical for alerting when predictions are uncertain. Sequential updating as new sensor data arrives.'
                },
                {
                    icon: '📊',
                    title: 'A/B Testing & Marketing',
                    description: 'Make decisions about campaigns with limited data. Bayesian approach provides probability distributions over conversion rates, enabling better decision-making with uncertainty quantification.'
                }
            ]
        },

        prosAndCons: {
            title: 'Advantages & Limitations',
            pros: [
                'Uncertainty Quantification: Provides full probability distributions for predictions',
                'Small Data Friendly: Works well with limited samples through informative priors',
                'Natural Regularization: Prior distributions automatically regularize',
                'Incorporates Prior Knowledge: Use domain expertise to set priors',
                'Sequential Learning: Can update beliefs as new data arrives',
                'Interpretable Uncertainty: Credible intervals have clear probabilistic interpretation',
                'Handles Underdetermined Systems: Works when features > samples',
                'No Overfitting Issues: Regularization through priors prevents overfitting'
            ],
            cons: [
                'Computational Cost: More expensive than OLS, especially with large datasets',
                'Prior Sensitivity: Results can depend heavily on prior choice',
                'Complex Implementation: More complex than frequentist methods',
                'Slower Inference: Prediction is slower due to integration',
                'Prior Specification: Requires thoughtful choice of priors',
                'Limited Scalability: Does not scale well to millions of samples',
                'Mathematical Complexity: Requires understanding of probability theory'
            ]
        },

        stepByStep: {
            title: 'Step-by-Step Algorithm',
            steps: [
                {
                    title: 'Specify Prior Distribution',
                    description: 'Define prior beliefs about parameters β ~ N(μ₀, Σ₀). Common choices: uninformative (μ₀=0, Σ₀=∞I) or informative based on domain knowledge. Prior encodes regularization.'
                },
                {
                    title: 'Define Likelihood',
                    description: 'Specify how data is generated: y = Xβ + ε where ε ~ N(0, σ²I). This assumes Gaussian noise with variance σ².'
                },
                {
                    title: 'Compute Posterior Distribution',
                    description: 'Apply Bayes theorem: P(β|y,X) ∝ P(y|X,β) × P(β). For Gaussian prior and likelihood, posterior is also Gaussian (conjugate prior).'
                },
                {
                    title: 'Calculate Posterior Mean',
                    description: 'Posterior mean: μₙ = (Σ₀⁻¹ + (1/σ²)XᵀX)⁻¹ × (Σ₀⁻¹μ₀ + (1/σ²)Xᵀy). This is the most likely parameter value given data.'
                },
                {
                    title: 'Calculate Posterior Covariance',
                    description: 'Posterior covariance: Σₙ = (Σ₀⁻¹ + (1/σ²)XᵀX)⁻¹. This quantifies uncertainty in parameter estimates.'
                },
                {
                    title: 'Make Predictions',
                    description: 'For new point x*, predictive distribution: y* ~ N(x*ᵀμₙ, x*ᵀΣₙx* + σ²). Mean is prediction, variance quantifies uncertainty.'
                },
                {
                    title: 'Compute Credible Intervals',
                    description: 'Extract 95% credible intervals from predictive distribution. These show where we expect 95% of predictions to fall.'
                },
                {
                    title: 'Update with New Data',
                    description: 'When new data arrives, posterior becomes new prior and repeat steps 3-6. This enables sequential/online learning.'
                }
            ]
        }
    },

    math: {
        objectiveFunction: {
            title: 'Bayesian Framework',
            formula: 'Posterior: P(β|y,X) = P(y|X,β) × P(β) / P(y|X)',
            parameters: [
                'P(β|y,X): Posterior distribution (what we want)',
                'P(y|X,β): Likelihood (how data is generated)',
                'P(β): Prior distribution (initial beliefs)',
                'P(y|X): Evidence (normalizing constant)',
                'β: Parameter vector (p × 1)',
                'y: Target vector (n × 1)',
                'X: Feature matrix (n × p)'
            ]
        },
        additionalContent: [
            {
                title: 'Prior Distribution',
                formula: 'β ~ N(μ₀, Σ₀)',
                description: 'Gaussian prior on parameters. μ₀ is prior mean (often 0), Σ₀ is prior covariance (controls regularization strength).'
            },
            {
                title: 'Likelihood Function',
                formula: 'P(y|X,β,σ²) = N(Xβ, σ²I)',
                description: 'Assumes Gaussian noise with variance σ². This is equivalent to standard linear regression assumption.'
            },
            {
                title: 'Posterior Distribution (Conjugate)',
                formula: 'β|y,X ~ N(μₙ, Σₙ)',
                description: 'Posterior is Gaussian when prior is Gaussian (conjugate prior property).'
            },
            {
                title: 'Posterior Mean',
                formula: 'μₙ = Σₙ(Σ₀⁻¹μ₀ + σ⁻²Xᵀy)',
                description: 'Weighted average of prior mean and maximum likelihood estimate.'
            },
            {
                title: 'Posterior Covariance',
                formula: 'Σₙ = (Σ₀⁻¹ + σ⁻²XᵀX)⁻¹',
                description: 'Combines prior uncertainty and data informativeness.'
            },
            {
                title: 'Predictive Distribution',
                formula: 'y*|x*,y,X ~ N(x*ᵀμₙ, σₚ²) where σₚ² = x*ᵀΣₙx* + σ²',
                description: 'Distribution of predictions. Variance includes parameter uncertainty (x*ᵀΣₙx*) and noise (σ²).'
            }
        ],
        visualization: {
            title: 'Bayesian Updating Visualization',
            items: [
                {
                    title: 'Prior',
                    color: '#94a3b8',
                    description: 'Initial beliefs before seeing data. Wide distribution indicates high uncertainty. Can be uninformative (flat) or informative (peaked).'
                },
                {
                    title: 'Likelihood',
                    color: '#3b82f6',
                    description: 'Information from observed data. Shape determined by data fit. Peaks at maximum likelihood estimate.'
                },
                {
                    title: 'Posterior',
                    color: '#8b5cf6',
                    description: 'Updated beliefs after seeing data. Compromise between prior and likelihood. Narrower than prior (reduced uncertainty).'
                }
            ],
            insight: 'As more data is observed, the posterior becomes increasingly peaked (more certain) and moves toward the true parameter value. The prior has less influence with more data. With infinite data, Bayesian and frequentist approaches converge.'
        }
    },

    code: {
        examples: [
            {
                title: 'Python (scikit-learn)',
                code: `from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Prepare data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features (recommended)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Method 1: Basic Bayesian Ridge Regression
bayesian_model = BayesianRidge(
    n_iter=300,
    tol=1e-3,
    alpha_1=1e-6,
    alpha_2=1e-6,
    lambda_1=1e-6,
    lambda_2=1e-6,
    compute_score=True
)

bayesian_model.fit(X_train_scaled, y_train)

# Make predictions with uncertainty
y_pred, y_std = bayesian_model.predict(X_test_scaled, return_std=True)

# 95% credible intervals
confidence_level = 1.96
y_lower = y_pred - confidence_level * y_std
y_upper = y_pred + confidence_level * y_std

print(f"Prediction: {y_pred[0]:.2f}")
print(f"95% Credible Interval: [{y_lower[0]:.2f}, {y_upper[0]:.2f}]")
print(f"Uncertainty (std): {y_std[0]:.2f}")`
            },
            {
                title: 'R (brms)',
                code: `library(brms)

# Fit Bayesian linear regression
bayesian_fit <- brm(
  formula = y ~ .,
  data = train_data,
  family = gaussian(),
  prior = c(
    prior(normal(0, 10), class = Intercept),
    prior(normal(0, 2.5), class = b),
    prior(cauchy(0, 1), class = sigma)
  ),
  chains = 4,
  iter = 2000
)

# Make predictions with credible intervals
predictions <- predict(bayesian_fit, newdata = test_data, probs = c(0.025, 0.975))
print(head(predictions))`
            },
            {
                title: 'Python (From Scratch)',
                code: `import numpy as np

class BayesianLinearRegression:
    def __init__(self, alpha=1.0, beta=1.0):
        self.alpha = alpha
        self.beta = beta
        self.mu_n = None
        self.Sigma_n = None
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Prior
        Sigma_0_inv = self.alpha * np.eye(n_features)
        mu_0 = np.zeros(n_features)
        
        # Posterior covariance
        self.Sigma_n = np.linalg.inv(
            Sigma_0_inv + self.beta * X.T @ X
        )
        
        # Posterior mean
        self.mu_n = self.Sigma_n @ (
            Sigma_0_inv @ mu_0 + self.beta * X.T @ y
        )
        
        return self
    
    def predict(self, X, return_std=False):
        y_pred = X @ self.mu_n
        
        if return_std:
            y_var = 1/self.beta + np.sum(X @ self.Sigma_n * X, axis=1)
            y_std = np.sqrt(y_var)
            return y_pred, y_std
        
        return y_pred

# Example usage
model = BayesianLinearRegression(alpha=1.0, beta=25.0)
model.fit(X_train, y_train)
y_pred, y_std = model.predict(X_test, return_std=True)`
            }
        ]
    },

    preprocessing: {
        critical: {
            title: 'Critical: Feature Scaling',
            description: 'While not strictly required, scaling is HIGHLY RECOMMENDED for Bayesian regression. It makes prior specification easier and improves numerical stability.',
            code: `from sklearn.preprocessing import StandardScaler\n\nscaler = StandardScaler()\nX_train_scaled = scaler.fit_transform(X_train)\nX_test_scaled = scaler.transform(X_test)`,
            why: 'When features have different scales, specifying a common prior becomes difficult. A prior that is reasonable for one feature might be completely wrong for another. Scaling puts all features on the same scale, making prior specification straightforward.'
        },
        stepsTitle: 'Recommended Preprocessing Pipeline',
        steps: [
            {
                title: 'Handle Missing Values',
                description: 'Bayesian methods can theoretically handle missing data, but most implementations require complete data.',
                code: `from sklearn.impute import SimpleImputer\n\nimputer = SimpleImputer(strategy='median')\nX_imputed = imputer.fit_transform(X)`
            },
            {
                title: 'Remove or Transform Outliers',
                description: 'Bayesian regression with Gaussian likelihood is sensitive to outliers. Consider robust alternatives or transform data.',
                code: `from sklearn.preprocessing import RobustScaler\n\nrobust_scaler = RobustScaler()\nX_robust = robust_scaler.fit_transform(X)`
            },
            {
                title: 'Encode Categorical Variables',
                description: 'Convert categorical features before fitting.',
                code: `from sklearn.preprocessing import OneHotEncoder\n\nencoder = OneHotEncoder(drop='first', sparse=False)\nX_encoded = encoder.fit_transform(X_categorical)`
            },
            {
                title: 'Standardization',
                description: 'Standardize features to mean=0, std=1. Makes prior specification uniform across features.',
                code: `from sklearn.preprocessing import StandardScaler\n\nscaler = StandardScaler()\nX_scaled = scaler.fit_transform(X)`
            }
        ],
        completePipeline: {
            title: 'Complete Preprocessing Pipeline',
            code: `from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import BayesianRidge

numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

full_pipeline = Pipeline([
    ('preprocessor', numerical_pipeline),
    ('bayesian', BayesianRidge(n_iter=300, compute_score=True))
])

full_pipeline.fit(X_train, y_train)
y_pred, y_std = full_pipeline.named_steps['bayesian'].predict(
    full_pipeline.named_steps['preprocessor'].transform(X_test),
    return_std=True
)`
        },
        mistakes: {
            title: 'Common Preprocessing Mistakes',
            items: [
                'Not scaling features - makes prior specification difficult',
                'Using same prior scale for different magnitude features',
                'Ignoring outliers - Gaussian likelihood is sensitive to them',
                'Forgetting that categorical encoding changes feature count',
                'Not accounting for missing data in prior specification',
                'Applying transforms that break Gaussian assumptions'
            ]
        }
    },

    tips: {
        hyperparameterTuning: {
            title: 'Hyperparameter Tuning & Prior Selection',
            sections: [
                {
                    title: 'Choosing Prior Parameters (α - alpha)',
                    points: [
                        'Alpha controls prior precision (1/variance) of weights',
                        'Small alpha (e.g., 1e-6) = weak prior = flexible model',
                        'Large alpha (e.g., 10) = strong prior = heavy regularization',
                        'BayesianRidge automatically learns alpha from data',
                        'Start with alpha=1.0 and let the model optimize'
                    ]
                },
                {
                    title: 'Noise Precision (β - lambda)',
                    points: [
                        'Lambda controls noise precision (1/noise_variance)',
                        'Small lambda = high noise = less trust in data',
                        'Large lambda = low noise = more trust in data',
                        'BayesianRidge learns lambda automatically',
                        'If you know noise level σ², set lambda = 1/σ²'
                    ]
                }
            ]
        },
        bestPractices: {
            dos: [
                'Always return uncertainty estimates',
                'Visualize credible intervals alongside predictions',
                'Check posterior distributions to ensure they are reasonable',
                'Use informative priors when you have domain knowledge',
                'Scale features before specifying priors',
                'Monitor convergence in iterative methods',
                'Use posterior predictive checks to validate model'
            ],
            donts: [
                'Do not ignore uncertainty estimates in decisions',
                'Do not use overly strong priors without justification',
                'Do not forget to check if posterior makes sense',
                'Do not compare predictions without comparing uncertainties',
                'Do not use with very large datasets',
                'Do not blindly trust default priors'
            ]
        },
        advancedTechniques: [
            {
                title: 'Sequential/Online Learning',
                description: 'Update model as new data arrives without retraining from scratch',
                code: `model.fit(X_batch1, y_batch1)\nX_combined = np.vstack([X_batch1, X_batch2])\ny_combined = np.concatenate([y_batch1, y_batch2])\nmodel.fit(X_combined, y_combined)`
            },
            {
                title: 'Visualize Uncertainty',
                description: 'Plot predictions with credible intervals',
                code: `import matplotlib.pyplot as plt\n\ny_pred, y_std = model.predict(X_test, return_std=True)\nplt.scatter(X_test[:, 0], y_test, alpha=0.5)\nplt.plot(X_test[:, 0], y_pred, 'r-')\nplt.fill_between(X_test[:, 0], y_pred - 1.96*y_std, y_pred + 1.96*y_std, alpha=0.2)`
            },
            {
                title: 'Model Comparison with Evidence',
                description: 'Use log marginal likelihood to compare models',
                code: `model.fit(X_train, y_train)\nevidence = model.scores_[-1]\nprint(f"Evidence = {evidence:.2f}")`
            }
        ],
        performance: {
            title: 'Performance Optimization',
            tips: [
                'For large datasets: Consider stochastic variational inference (SVI)',
                'Use sparse matrices when applicable',
                'For real-time: Pre-compute posterior and use only predict()',
                'Parallel chains: Use multiple MCMC chains in parallel',
                'Approximate methods: Variational inference faster than MCMC',
                'Batch processing: Process predictions in batches for efficiency'
            ]
        },
        debugging: {
            title: 'Debugging Common Issues',
            issues: [
                {
                    problem: 'Problem: Very wide credible intervals (high uncertainty)',
                    solution: 'Either need more data, features are not informative, or prior is too weak. Try more data or stronger prior.'
                },
                {
                    problem: 'Problem: Model not converging',
                    solution: 'Increase n_iter, check if features are scaled, reduce tol parameter. Check model.scores_ for convergence pattern.'
                },
                {
                    problem: 'Problem: Posterior is very different from prior',
                    solution: 'This is often good - means data is informative! But if unexpected, check data quality and prior specification.'
                },
                {
                    problem: 'Problem: Predictions identical to OLS',
                    solution: 'Prior is too weak or data overwhelms prior. This happens with large datasets - Bayesian and frequentist converge.'
                },
                {
                    problem: 'Problem: Numerical instability / NaN values',
                    solution: 'Scale features, check for extreme outliers, increase regularization (larger alpha), check condition number of XᵀX.'
                },
                {
                    problem: 'Problem: Uncertainty too small (overconfident)',
                    solution: 'Noise parameter (lambda) might be too large. Check if beta/lambda makes sense for your problem scale.'
                },
                {
                    problem: 'Problem: Credible intervals do not contain true values',
                    solution: 'Model misspecification - check linearity assumption, add polynomial features, or consider non-linear model.'
                }
            ]
        }
    }
};
