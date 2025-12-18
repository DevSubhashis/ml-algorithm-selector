import React, { useState } from 'react';
import { ChevronRight, ChevronLeft, Info, CheckCircle, BookOpen, ExternalLink } from 'lucide-react';
import AlgorithmDetailPage from './AlgorithmDetailPage';
import { elasticNetData } from '../data/elasticNetData';
import { bayesianLinearData } from '../data/bayesianLinearData';
import { robustRegressionData } from '../data/robustRegressionData';
import { lassoRegressionData } from '../data/lassoRegressionData';
import { ridgeRegressionData } from '../data/ridgeRegressionData';
import { olsRegressionData } from '../data/olsRegressionData';

const MLAlgorithmSelector = () => {
  const [showDetailPage, setShowDetailPage] = useState(null);
  
  // Map algorithm keys to their data
  const algorithmDataMap = {
    'elasticnet': elasticNetData,
    'bayesianlinearregression': bayesianLinearData,
    'robustregression': robustRegressionData,
    'huberregression': robustRegressionData,
    'ransac': robustRegressionData,
    'theilsenestimator': robustRegressionData,
    'lassoregressionl1': lassoRegressionData,
    'lasso': lassoRegressionData,
    'ridgeregressionl2': ridgeRegressionData,
    'ridge': ridgeRegressionData,
    'ordinaryleastsquaresols': olsRegressionData,
    'ols': olsRegressionData,
    'linearregression': olsRegressionData,
  };

  const [step, setStep] = useState(0);
  const [answers, setAnswers] = useState({});

  const questions = [
    {
      id: 'dataLabels',
      question: 'Do you have labeled data?',
      info: 'Labeled data means each training example has an associated output/target value. For example, images tagged as "cat" or "dog", or house prices with their features.',
      options: [
        { value: 'yes', label: 'Yes - I have input-output pairs', next: 'outputType' },
        { value: 'no', label: 'No - Only input data', next: 'unsupervisedTask' },
        { value: 'partial', label: 'Partial - Some labeled, mostly unlabeled', next: 'semiSupervised' }
      ]
    },
    {
      id: 'outputType',
      question: 'What type of output do you need?',
      info: 'The type of prediction you want determines the algorithm category.',
      options: [
        { value: 'continuous', label: 'Continuous values (e.g., prices, temperatures)', next: 'regressionSize' },
        { value: 'categorical', label: 'Categories/Classes (e.g., spam/not spam)', next: 'numClasses' },
        { value: 'sequence', label: 'Sequences (e.g., text, time series)', next: 'sequenceType' },
        { value: 'ranking', label: 'Rankings (e.g., search results)', next: 'rankingAlgo' },
        { value: 'recommendation', label: 'Recommendations', next: 'recommendationType' }
      ]
    },
    {
      id: 'regressionSize',
      question: 'How much data do you have and what are its characteristics?',
      info: 'Data size and complexity significantly impact algorithm choice.',
      options: [
        { value: 'small-linear', label: 'Small dataset (<1K samples), linear relationships', next: 'linearRegression' },
        { value: 'small-nonlinear', label: 'Small dataset, non-linear relationships', next: 'nonlinearSmall' },
        { value: 'outliers', label: 'Dataset contains outliers or noisy data', next: 'robustRegression' },
        { value: 'medium', label: 'Medium dataset (1K-100K samples)', next: 'mediumRegression' },
        { value: 'large', label: 'Large dataset (>100K samples)', next: 'largeRegression' },
        { value: 'time-series', label: 'Time series data', next: 'timeSeriesRegression' }
      ]
    },
    {
      id: 'numClasses',
      question: 'How many classes/categories?',
      info: 'The number of output classes affects algorithm selection.',
      options: [
        { value: 'binary', label: 'Two classes (Binary Classification)', next: 'binaryFeatures' },
        { value: 'multiclass', label: 'Multiple classes (3-100)', next: 'multiclassType' },
        { value: 'multilabel', label: 'Multiple labels per sample', next: 'multilabelAlgo' },
        { value: 'many', label: 'Very many classes (>100)', next: 'manyClassAlgo' }
      ]
    },
    {
      id: 'binaryFeatures',
      question: 'What are your data characteristics and priorities?',
      info: 'Different algorithms excel with different data types and priorities.',
      options: [
        { value: 'small-interpretable', label: 'Small data, need interpretability', next: 'logisticRegression' },
        { value: 'text', label: 'Text data', next: 'textClassification' },
        { value: 'images', label: 'Image data', next: 'imageClassification' },
        { value: 'tabular-small', label: 'Tabular data, <10K samples', next: 'smallTabular' },
        { value: 'tabular-large', label: 'Tabular data, >10K samples', next: 'largeTabular' },
        { value: 'imbalanced', label: 'Highly imbalanced classes', next: 'imbalancedClass' }
      ]
    },
    {
      id: 'multiclassType',
      question: 'What are your requirements?',
      info: 'Different algorithms offer different trade-offs.',
      options: [
        { value: 'interpretable', label: 'Need interpretability', next: 'interpretableMulti' },
        { value: 'performance', label: 'Maximum performance, any complexity', next: 'performanceMulti' },
        { value: 'fast', label: 'Fast training and prediction', next: 'fastMulti' },
        { value: 'probabilistic', label: 'Need probability estimates', next: 'probabilisticMulti' }
      ]
    },
    {
      id: 'unsupervisedTask',
      question: 'What is your goal with unlabeled data?',
      info: 'Unsupervised learning has several different objectives.',
      options: [
        { value: 'clustering', label: 'Find natural groups/clusters', next: 'clusteringType' },
        { value: 'dimreduce', label: 'Reduce dimensions/visualize', next: 'dimReductionType' },
        { value: 'anomaly', label: 'Detect anomalies/outliers', next: 'anomalyType' },
        { value: 'association', label: 'Find association rules/patterns', next: 'associationAlgo' },
        { value: 'density', label: 'Estimate probability density', next: 'densityAlgo' },
        { value: 'generation', label: 'Generate new samples', next: 'generativeType' }
      ]
    },
    {
      id: 'clusteringType',
      question: 'What do you know about your clusters?',
      info: 'Cluster characteristics guide algorithm selection.',
      options: [
        { value: 'spherical', label: 'Spherical clusters, know number', next: 'kmeans' },
        { value: 'arbitrary', label: 'Arbitrary shapes, unknown number', next: 'dbscan' },
        { value: 'hierarchical', label: 'Hierarchical structure', next: 'hierarchical' },
        { value: 'probabilistic', label: 'Probabilistic/soft clustering', next: 'gmm' },
        { value: 'large', label: 'Very large dataset', next: 'largeClustering' }
      ]
    },
    {
      id: 'dimReductionType',
      question: 'What is your goal?',
      info: 'Dimensionality reduction serves different purposes.',
      options: [
        { value: 'linear', label: 'Linear projection, preserve variance', next: 'pca' },
        { value: 'visualization', label: 'Visualization (2D/3D)', next: 'visualization' },
        { value: 'supervised', label: 'Supervised dimension reduction', next: 'lda' },
        { value: 'sparse', label: 'Sparse features/feature selection', next: 'sparseMethod' },
        { value: 'nonlinear', label: 'Non-linear manifold learning', next: 'manifold' }
      ]
    },
    {
      id: 'sequenceType',
      question: 'What type of sequence task?',
      info: 'Sequence modeling has evolved significantly with deep learning.',
      options: [
        { value: 'text-classification', label: 'Text classification', next: 'textSeqClass' },
        { value: 'text-generation', label: 'Text generation', next: 'textGeneration' },
        { value: 'translation', label: 'Machine translation', next: 'translation' },
        { value: 'time-series', label: 'Time series forecasting', next: 'timeSeriesForecast' },
        { value: 'speech', label: 'Speech recognition', next: 'speechRecog' }
      ]
    },
    {
      id: 'semiSupervised',
      question: 'What is the ratio of labeled to unlabeled data?',
      info: 'Semi-supervised learning leverages both labeled and unlabeled data.',
      options: [
        { value: 'very-few', label: '<1% labeled', next: 'semiSupervisedAlgo' },
        { value: 'few', label: '1-10% labeled', next: 'semiSupervisedAlgo' },
        { value: 'moderate', label: '10-30% labeled', next: 'semiSupervisedAlgo' }
      ]
    },
    {
      id: 'anomalyType',
      question: 'What type of anomaly detection?',
      info: 'Different scenarios require different anomaly detection approaches.',
      options: [
        { value: 'supervised', label: 'Have labeled anomalies', next: 'supervisedAnomaly' },
        { value: 'unsupervised', label: 'No labeled anomalies', next: 'unsupervisedAnomaly' },
        { value: 'online', label: 'Real-time/streaming data', next: 'onlineAnomaly' }
      ]
    }
  ];

  const algorithms = {
    // Linear Regression Family
    linearRegression: {
      name: 'Linear Regression Family',
      algorithms: ['Ordinary Least Squares (OLS)', 'Ridge Regression (L2)', 'Lasso Regression (L1)', 'Elastic Net', 'Bayesian Linear Regression', 'Robust Regression'],
      why: 'You selected small dataset with linear relationships. Linear regression is ideal because:\n\n• Simple and interpretable - easy to explain coefficients\n• Fast to train and predict\n• Works well with limited data\n• OLS is the foundation - BLUE estimator\n• Ridge/Lasso help with multicollinearity and feature selection\n• Elastic Net combines benefits of Ridge and Lasso\n• Bayesian version provides uncertainty estimates\n• Robust methods handle outliers',
      when: 'Use when relationships are linear, you need interpretability, or have limited data',
      pros: 'Fast, interpretable, well-understood theory, works with small data',
      cons: 'Cannot model non-linear relationships, sensitive to outliers (except robust), assumes linear relationships',
      details: 'OLS minimizes sum of squared errors. Ridge adds L2 penalty. Lasso adds L1 penalty (feature selection). Elastic Net combines both. Bayesian uses probabilistic framework. Robust methods downweight outliers.'
    },

    nonlinearSmall: {
      name: 'Non-linear Regression (Small Data)',
      algorithms: ['Polynomial Regression', 'Support Vector Regression (SVR)', 'Kernel Ridge Regression', 'Gaussian Process Regression', 'K-Nearest Neighbors Regression', 'Robust Regression (Huber, RANSAC)'],
      why: 'You have non-linear relationships with limited data. These algorithms handle non-linearity:\n\n• Polynomial Regression extends linear models with polynomial features\n• SVR uses kernel trick for non-linear mapping\n• Gaussian Processes provide probabilistic predictions with uncertainty\n• KNN makes predictions based on similar examples\n• Robust Regression handles outliers that might distort the fit\n• All work reasonably well with small datasets',
      when: 'Use when relationships are clearly non-linear but data is limited',
      pros: 'Can model complex relationships, some provide uncertainty estimates',
      cons: 'More hyperparameters to tune, can overfit with small data, slower than linear models',
      details: 'Polynomial features create x², x³ terms. SVR uses RBF or polynomial kernels. GPs model functions as distributions. KNN uses distance metrics. Robust methods downweight outliers.'
    },

    robustRegression: {
      name: 'Robust Regression',
      algorithms: ['Huber Regression', 'RANSAC', 'Theil-Sen Estimator', 'LAD (L1) Regression'],
      why: 'You have data with outliers or noise. Robust regression is designed for this:\n\n• Huber Regression: Uses robust loss function, downweights large residuals\n• RANSAC: Randomly samples inliers, ignores outliers completely\n• Theil-Sen: Median-based, can handle up to 29% outliers\n• LAD Regression: Minimizes absolute errors instead of squared\n• All resistant to outliers that would ruin OLS\n• No need to manually remove outliers',
      when: 'Use when data quality is questionable, contains outliers, or has measurement errors',
      pros: 'Outlier resistant, stable estimates, automatic outlier handling, preserves information',
      cons: 'Slower than OLS, requires parameter tuning, less efficient on clean data',
      details: 'Huber uses piecewise loss function. RANSAC fits to random subsets and finds consensus. Theil-Sen computes median of all pairwise slopes. LAD minimizes |y - ŷ|.'
    },

    mediumRegression: {
      name: 'Tree-Based Regression',
      algorithms: ['Random Forest', 'Gradient Boosting (XGBoost, LightGBM, CatBoost)', 'Extra Trees', 'Histogram-based Gradient Boosting'],
      why: 'You have medium-sized data. Tree-based methods excel here:\n\n• Random Forest: Ensemble of trees, reduces variance, handles non-linearity\n• Gradient Boosting: Sequential trees, high accuracy, XGBoost/LightGBM are optimized versions\n• CatBoost: Handles categorical features natively\n• Extra Trees: Similar to RF but faster\n• All handle mixed feature types and missing values well',
      when: 'Use with tabular data, mixed feature types, when you need good performance without deep learning',
      pros: 'High accuracy, handles non-linearity, requires little preprocessing, feature importance, handles missing values',
      cons: 'Can overfit, slower than linear models, less interpretable, memory intensive',
      details: 'RF averages multiple trees. GBM builds trees sequentially correcting previous errors. XGBoost uses regularization and parallel processing. LightGBM uses histogram-based splitting.'
    },

    largeRegression: {
      name: 'Large-Scale Regression',
      algorithms: ['Neural Networks (MLPs)', 'Deep Learning (ResNets for structured data)', 'Online/Stochastic Gradient Descent variants', 'XGBoost/LightGBM (optimized for large data)', 'Linear Models with feature hashing'],
      why: 'You have large-scale data (>100K samples). These scale well:\n\n• Neural Networks: Can learn complex patterns, scale with data\n• Deep Learning: Multiple layers for hierarchical features\n• SGD variants: Update on mini-batches, memory efficient\n• LightGBM: Optimized gradient boosting for large data\n• Feature hashing: Reduces memory for high-dimensional data',
      when: 'Use with very large datasets where accuracy improves with more data',
      pros: 'Scale to massive data, can learn very complex patterns, state-of-the-art performance',
      cons: 'Require significant compute, need careful tuning, less interpretable, need more data to avoid overfitting',
      details: 'MLPs have multiple hidden layers. SGD processes mini-batches. LightGBM uses histogram binning and leaf-wise growth. Feature hashing uses hash functions for dimensionality reduction.'
    },

    timeSeriesRegression: {
      name: 'Time Series Forecasting',
      algorithms: ['ARIMA/SARIMA', 'Exponential Smoothing (Holt-Winters)', 'Prophet (Facebook)', 'LSTM/GRU', 'Temporal Convolutional Networks', 'Transformer models (Temporal Fusion Transformer)', 'N-BEATS', 'DeepAR'],
      why: 'You have time series data with temporal dependencies:\n\n• ARIMA: Classical statistical method, good for stationary series\n• Prophet: Handles seasonality, holidays, missing data automatically\n• LSTM/GRU: Deep learning, learns long-term dependencies\n• Transformers: Attention mechanism, state-of-the-art for many tasks\n• N-BEATS: Pure deep learning architecture, interpretable\n• DeepAR: Probabilistic forecasting at scale',
      when: 'Use when data has temporal structure, seasonality, or trends',
      pros: 'Handles temporal patterns, seasonality, and trends naturally',
      cons: 'Require sufficient history, sensitive to non-stationarity, deep models need lots of data',
      details: 'ARIMA uses autoregression and moving averages. LSTM uses memory cells. Transformers use self-attention. Prophet decomposes into trend, seasonality, and holidays.'
    },

    logisticRegression: {
      name: 'Logistic Regression Family',
      algorithms: ['Logistic Regression (L2)', 'Logistic Regression (L1)', 'Elastic Net Logistic', 'Multinomial Logistic'],
      why: 'You need interpretability with small data:\n\n• Linear decision boundary, easy to understand\n• Coefficients show feature importance and direction\n• Provides probability estimates\n• Fast training and prediction\n• L1 regularization for feature selection\n• Works well with limited samples\n• Industry standard for credit scoring, medical diagnosis',
      when: 'Use when you need to explain decisions to stakeholders or regulators',
      pros: 'Highly interpretable, fast, provides probabilities, works with small data, no hyperparameters',
      cons: 'Cannot learn non-linear patterns, assumes linear separability',
      details: 'Uses logistic function to model probabilities. L2 adds weight decay. L1 can zero out features. Multinomial extends to multiple classes.'
    },
  };

  const currentQuestion = questions.find(q => q.id === (step === 0 ? 'dataLabels' : answers[questions[step - 1]?.id]?.next || 'dataLabels'));
  const result = algorithms[answers[currentQuestion?.id]?.next];

  // Show detail page if requested
  if (showDetailPage && algorithmDataMap[showDetailPage]) {
    return (
      <AlgorithmDetailPage
        algorithmData={algorithmDataMap[showDetailPage]}
        onBack={() => setShowDetailPage(null)}
      />
    );
  }

  const handleAnswer = (option) => {
    setAnswers({ ...answers, [currentQuestion.id]: option });
    if (option.next && !algorithms[option.next]) {
      setStep(step + 1);
    }
  };

  const handleBack = () => {
    if (step > 0) {
      const prevQuestions = questions.slice(0, step);
      const lastAnsweredId = prevQuestions.reverse().find(q => answers[q.id])?.id;
      if (lastAnsweredId) {
        const newAnswers = { ...answers };
        delete newAnswers[lastAnsweredId];
        setAnswers(newAnswers);
      }
      setStep(step - 1);
    }
  };

  const handleReset = () => {
    setAnswers({});
    setStep(0);
    setShowDetailPage(null);
  };

  if (result) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-8">
        <div className="max-w-4xl mx-auto">
          <div className="bg-white rounded-2xl shadow-2xl p-8">
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center gap-3">
                <CheckCircle className="w-8 h-8 text-green-500" />
                <h1 className="text-3xl font-bold text-gray-800">Recommended Algorithm</h1>
              </div>
              <button
                onClick={handleReset}
                className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition"
              >
                Start Over
              </button>
            </div>

            <div className="mb-6 p-4 bg-indigo-50 rounded-lg border-l-4 border-indigo-600">
              <h2 className="text-2xl font-semibold text-indigo-900 mb-2">{result.name}</h2>
              <div className="flex flex-wrap gap-2 mt-3">
                {result.algorithms.map((algo, idx) => {
                  const algoKey = algo.toLowerCase().replace(/[^a-z0-9]/g, '');
                  const hasDetailPage = ['elasticnet', 'bayesianlinearregression', 'robustregression', 'huberregression', 'ransac', 'theilsenestimator', 'lassoregressionl1', 'lasso', 'ridgeregressionl2', 'ridge', 'ordinaryleastsquaresols', 'ols', 'linearregression'].includes(algoKey);
                  
                  return (
                    <button
                      key={idx}
                      onClick={() => hasDetailPage && setShowDetailPage(algoKey)}
                      className={`px-3 py-1 bg-white rounded-full text-sm font-medium text-indigo-700 border border-indigo-200 ${
                        hasDetailPage ? 'hover:bg-indigo-100 cursor-pointer flex items-center gap-1' : ''
                      }`}
                      disabled={!hasDetailPage}
                    >
                      {algo}
                      {hasDetailPage && <ExternalLink className="w-3 h-3" />}
                    </button>
                  );
                })}
              </div>
            </div>

            <div className="space-y-6">
              <div className="p-6 bg-blue-50 rounded-xl border border-blue-200">
                <h3 className="text-xl font-semibold text-blue-900 mb-3 flex items-center gap-2">
                  <Info className="w-5 h-5" />
                  Why This Algorithm?
                </h3>
                <p className="text-gray-700 whitespace-pre-line leading-relaxed">{result.why}</p>
              </div>

              <div className="grid md:grid-cols-2 gap-4">
                <div className="p-5 bg-green-50 rounded-xl border border-green-200">
                  <h4 className="font-semibold text-green-900 mb-2">✓ Advantages</h4>
                  <p className="text-gray-700 text-sm">{result.pros}</p>
                </div>
                <div className="p-5 bg-red-50 rounded-xl border border-red-200">
                  <h4 className="font-semibold text-red-900 mb-2">✗ Limitations</h4>
                  <p className="text-gray-700 text-sm">{result.cons}</p>
                </div>
              </div>

              <div className="p-6 bg-purple-50 rounded-xl border border-purple-200">
                <h4 className="font-semibold text-purple-900 mb-2 flex items-center gap-2">
                  <BookOpen className="w-5 h-5" />
                  When to Use
                </h4>
                <p className="text-gray-700 text-sm">{result.when}</p>
              </div>

              <div className="p-6 bg-gray-50 rounded-xl border border-gray-200">
                <h4 className="font-semibold text-gray-900 mb-2">Technical Details</h4>
                <p className="text-gray-700 text-sm">{result.details}</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-8">
      <div className="max-w-3xl mx-auto">
        <div className="bg-white rounded-2xl shadow-2xl p-8">
          <div className="mb-8">
            <h1 className="text-3xl font-bold text-gray-800 mb-2">ML Algorithm Selector</h1>
            <p className="text-gray-600">Answer questions to find the perfect machine learning algorithm</p>
            <div className="mt-4 flex gap-2">
              {Array.from({ length: 5 }).map((_, idx) => (
                <div
                  key={idx}
                  className={`h-2 flex-1 rounded-full ${
                    idx <= step ? 'bg-indigo-600' : 'bg-gray-200'
                  }`}
                />
              ))}
            </div>
          </div>

          {currentQuestion && (
            <div>
              <div className="mb-6 p-4 bg-indigo-50 rounded-lg border-l-4 border-indigo-600">
                <h2 className="text-xl font-semibold text-gray-800 mb-2">{currentQuestion.question}</h2>
                <p className="text-sm text-gray-600 flex items-start gap-2">
                  <Info className="w-4 h-4 mt-0.5 flex-shrink-0" />
                  {currentQuestion.info}
                </p>
              </div>

              <div className="space-y-3 mb-6">
                {currentQuestion.options.map((option, idx) => (
                  <button
                    key={idx}
                    onClick={() => handleAnswer(option)}
                    className="w-full p-4 text-left bg-white border-2 border-gray-200 rounded-xl hover:border-indigo-500 hover:bg-indigo-50 transition duration-200 flex items-center justify-between group"
                  >
                    <span className="text-gray-800 font-medium">{option.label}</span>
                    <ChevronRight className="w-5 h-5 text-gray-400 group-hover:text-indigo-600 transition" />
                  </button>
                ))}
              </div>

              {step > 0 && (
                <button
                  onClick={handleBack}
                  className="flex items-center gap-2 text-indigo-600 hover:text-indigo-700 font-medium"
                >
                  <ChevronLeft className="w-5 h-5" />
                  Go Back
                </button>
              )}
            </div>
          )}
        </div>

        <div className="mt-6 p-4 bg-white rounded-lg shadow text-sm text-gray-600">
          <p className="font-medium mb-2">💡 Tips:</p>
          <ul className="space-y-1 list-disc list-inside">
            <li>Be honest about your data size and characteristics</li>
            <li>Consider whether you need interpretability</li>
            <li>Think about computational resources available</li>
            <li>Start simple, then increase complexity if needed</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default MLAlgorithmSelector;