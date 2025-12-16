import React, { useState } from 'react';
import { ChevronRight, ChevronLeft, Info, CheckCircle, BookOpen, ExternalLink } from 'lucide-react';
import AlgorithmDetailPage from './AlgorithmDetailPage';
import { elasticNetData } from '../data/elasticNetData';
import { bayesianLinearData } from '../data/bayesianLinearData';
import { robustRegressionData } from '../data/robustRegressionData';
import { lassoRegressionData } from '../data/lassoRegressionData';
import { ridgeRegressionData } from '../data/ridgeRegressionData';

const MLAlgorithmSelector = () => {
  const [showDetailPage, setShowDetailPage] = useState(null);
  
  // Map algorithm keys to their data
  const algorithmDataMap = {
    'elasticnet': elasticNetData,
    'bayesianlinearregression': bayesianLinearData,
    'robustregression': robustRegressionData,
    'huberregression': robustRegressionData, 
    'ransac': robustRegressionData, 
    'lassoregressionl1': lassoRegressionData,
    'lasso': lassoRegressionData,  
    'ridgeregressionl2': ridgeRegressionData,
    'ridge' : ridgeRegressionData
    // Add more algorithms here as you create them
  };
  const [step, setStep] = useState(0);
  const [answers, setAnswers] = useState({});
  const [showExplanation, setShowExplanation] = useState(false);

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
      why: 'You selected small dataset with linear relationships. Linear regression is ideal because:\n\n• Simple and interpretable - easy to explain coefficients\n• Fast to train and predict\n• Works well with limited data\n• No hyperparameter tuning needed for OLS\n• Ridge/Lasso help with multicollinearity and feature selection\n• Elastic Net combines benefits of Ridge and Lasso\n• Bayesian version provides uncertainty estimates',
      when: 'Use when relationships are linear, you need interpretability, or have limited data',
      pros: 'Fast, interpretable, no hyperparameters, works with small data',
      cons: 'Cannot model non-linear relationships, sensitive to outliers, assumes linear relationships',
      details: 'OLS minimizes sum of squared errors. Ridge adds L2 penalty (shrinks coefficients). Lasso adds L1 penalty (can zero out features). Elastic Net combines both. Bayesian version uses probabilistic framework.'
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

    textClassification: {
      name: 'Text Classification',
      algorithms: ['Naive Bayes (Multinomial)', 'Logistic Regression with TF-IDF', 'Linear SVM', 'FastText', 'BERT/RoBERTa/DistilBERT', 'XLNet', 'GPT fine-tuned'],
      why: 'You have text data for classification:\n\n• Naive Bayes: Fast baseline, works well for spam detection\n• TF-IDF + Linear models: Classic approach, very fast\n• FastText: Efficient embeddings, handles large vocabularies\n• BERT family: State-of-the-art, understands context\n• Fine-tuned LLMs: Best performance, can do few-shot learning',
      when: 'Use for sentiment analysis, spam detection, topic classification',
      pros: 'BERT models understand context, FastText is very fast, Naive Bayes works with small data',
      cons: 'BERT models are large and slow, simpler models miss context',
      details: 'Naive Bayes uses word frequencies. TF-IDF weights words by importance. BERT uses bidirectional transformers with pre-training on massive text.'
    },

    imageClassification: {
      name: 'Image Classification',
      algorithms: ['Convolutional Neural Networks (CNNs)', 'ResNet/ResNeXt', 'EfficientNet', 'Vision Transformers (ViT)', 'CLIP', 'Transfer Learning (ImageNet pre-trained models)', 'MobileNet (for mobile/edge)', 'YOLO (if also need detection)'],
      why: 'You have image data:\n\n• CNNs are designed for spatial data, learn hierarchical features\n• ResNet solves vanishing gradient with skip connections\n• EfficientNet scales networks optimally\n• Vision Transformers: Recent breakthrough, attention-based\n• Transfer learning: Use pre-trained models, works with small data\n• MobileNet: Optimized for resource-constrained devices',
      when: 'Use for any image classification task',
      pros: 'State-of-the-art accuracy, transfer learning works with limited data, handles raw pixels',
      cons: 'Requires GPUs, lots of data for training from scratch, computationally expensive',
      details: 'CNNs use convolutional layers to detect patterns. ResNet uses skip connections. Transfer learning fine-tunes pre-trained models. ViT splits images into patches.'
    },

    smallTabular: {
      name: 'Small Tabular Data Classification',
      algorithms: ['Logistic Regression', 'Support Vector Machines (SVM)', 'Decision Trees', 'Random Forest', 'Naive Bayes', 'K-Nearest Neighbors (KNN)'],
      why: 'You have tabular data with <10K samples:\n\n• Logistic Regression: Fast, interpretable baseline\n• SVM with RBF kernel: Handles non-linear boundaries well\n• Decision Trees: Interpretable, handles mixed features\n• Random Forest: More robust than single tree\n• Naive Bayes: Fast, works well with small data\n• KNN: Simple, non-parametric',
      when: 'Use with structured/tabular data and limited samples',
      pros: 'Fast training, some are interpretable, work with small datasets',
      cons: 'May not capture complex patterns, need feature engineering',
      details: 'SVM finds maximum margin hyperplane. RF averages multiple trees. KNN predicts based on nearest neighbors. Naive Bayes assumes feature independence.'
    },

    largeTabular: {
      name: 'Large Tabular Data Classification',
      algorithms: ['XGBoost', 'LightGBM', 'CatBoost', 'Neural Networks', 'TabNet', 'Deep & Cross Networks', 'NODE (Neural Oblivious Decision Ensembles)'],
      why: 'You have large tabular dataset (>10K samples):\n\n• XGBoost: Industry standard, wins many Kaggle competitions\n• LightGBM: Faster than XGBoost, handles large data well\n• CatBoost: Handles categorical features natively, reduces overfitting\n• TabNet: Deep learning designed for tabular data\n• Neural networks can learn complex interactions',
      when: 'Use with large structured datasets for maximum performance',
      pros: 'State-of-the-art accuracy, handles complex interactions, built-in regularization',
      cons: 'Requires tuning, less interpretable, can overfit, slower than simpler models',
      details: 'XGBoost uses regularized boosting. LightGBM uses histogram-based splitting. CatBoost uses ordered boosting. TabNet uses sequential attention.'
    },

    imbalancedClass: {
      name: 'Imbalanced Classification',
      algorithms: ['SMOTE (Synthetic Minority Over-sampling)', 'Random Under/Over-sampling', 'Class-weighted models', 'Anomaly Detection approaches', 'Focal Loss with Neural Networks', 'EasyEnsemble', 'BalancedRandomForest'],
      why: 'You have imbalanced classes (e.g., 99% negative, 1% positive):\n\n• SMOTE: Generates synthetic minority samples\n• Class weights: Penalizes misclassifying minority class more\n• Anomaly detection: Treats minority as anomalies\n• Focal Loss: Focuses on hard examples\n• Ensemble methods: Combine multiple balanced models\n• Specialized evaluation metrics: Use F1, AUC-PR instead of accuracy',
      when: 'Use when one class is much rarer than others (fraud, disease detection)',
      pros: 'Improves minority class recall, various approaches available',
      cons: 'SMOTE can generate noise, class weights need tuning, may sacrifice majority class accuracy',
      details: 'SMOTE interpolates between minority samples. Class weights multiply loss by class weight. Focal Loss adds (1-p)^γ factor to focus on hard examples.'
    },

    interpretableMulti: {
      name: 'Interpretable Multi-class Classification',
      algorithms: ['Decision Trees', 'Random Forest', 'Naive Bayes', 'Logistic Regression (one-vs-rest)', 'Rule-based classifiers (RIPPER)', 'Linear Discriminant Analysis'],
      why: 'You need interpretable multi-class classification:\n\n• Decision Trees: Produces clear if-then rules\n• Random Forest: Feature importance, can extract rules\n• Naive Bayes: Simple probability model\n• Logistic Regression: Linear coefficients per class\n• Rule-based: Explicit human-readable rules\n• LDA: Projects to discriminative directions',
      when: 'Use when stakeholders need to understand why predictions were made',
      pros: 'Easy to explain, fast inference, debuggable',
      cons: 'May sacrifice accuracy for interpretability',
      details: 'Trees split on feature thresholds. One-vs-rest trains binary classifier per class. RIPPER learns decision rules. LDA finds linear combinations that separate classes.'
    },

    performanceMulti: {
      name: 'High-Performance Multi-class Classification',
      algorithms: ['XGBoost', 'LightGBM', 'CatBoost', 'Neural Networks', 'Support Vector Machines (one-vs-one)', 'Stacking Ensembles'],
      why: 'You prioritize maximum performance:\n\n• Gradient Boosting: Highest accuracy on tabular data\n• Neural Networks: Can learn complex patterns\n• SVM: Effective in high dimensions\n• Stacking: Combines multiple models for best results\n• These win competitions and production benchmarks',
      when: 'Use when accuracy is paramount and you have computational resources',
      pros: 'State-of-the-art accuracy, handles complex patterns',
      cons: 'Less interpretable, requires tuning, computationally expensive',
      details: 'Boosting builds sequential correcting trees. Neural nets use multiple layers. One-vs-one trains C(C-1)/2 binary classifiers. Stacking uses meta-learner on base predictions.'
    },

    fastMulti: {
      name: 'Fast Multi-class Classification',
      algorithms: ['Naive Bayes', 'Linear SVM', 'Logistic Regression', 'Linear Discriminant Analysis', 'Stochastic Gradient Descent'],
      why: 'You need fast training and prediction:\n\n• Naive Bayes: O(nd) training, instant prediction\n• Linear models: Fast matrix operations\n• SGD: Processes one sample at a time\n• No complex tree building or kernel computations\n• Suitable for real-time systems',
      when: 'Use for real-time prediction, large-scale systems, or resource-constrained environments',
      pros: 'Very fast, low memory, scales to large data',
      cons: 'Cannot model complex non-linear patterns',
      details: 'Naive Bayes multiplies probabilities. Linear models use matrix multiplication. SGD updates weights incrementally. LDA solves generalized eigenvalue problem.'
    },

    probabilisticMulti: {
      name: 'Probabilistic Multi-class Classification',
      algorithms: ['Naive Bayes', 'Logistic Regression (softmax)', 'Neural Networks (softmax output)', 'Gaussian Process Classification', 'Calibrated classifiers (Platt scaling, Isotonic)'],
      why: 'You need well-calibrated probability estimates:\n\n• Important for decision making under uncertainty\n• Medical diagnosis, risk assessment need probabilities\n• Naive Bayes directly models probabilities\n• Softmax produces probability distributions\n• GP provides uncertainty estimates\n• Calibration methods correct probability estimates',
      when: 'Use when you need confidence estimates, not just class labels',
      pros: 'Provides confidence, supports decision theory, can defer uncertain predictions',
      cons: 'Some classifiers need calibration, GPs are slow',
      details: 'Softmax exponentiates logits and normalizes. Platt scaling fits logistic regression on classifier outputs. Isotonic regression fits monotonic function.'
    },

    multilabelAlgo: {
      name: 'Multi-label Classification',
      algorithms: ['Binary Relevance', 'Classifier Chains', 'Label Powerset', 'Multi-label k-NN', 'Neural Networks with sigmoid outputs', 'ML-KNN'],
      why: 'You need to predict multiple labels per sample:\n\n• Binary Relevance: Train separate classifier per label\n• Classifier Chains: Models label dependencies\n• Label Powerset: Treats each label combination as class\n• Neural networks with sigmoid: Predicts each label independently\n• Multi-label KNN: Adapts KNN for multiple labels',
      when: 'Use for document tagging, image multi-tagging, multi-symptom diagnosis',
      pros: 'Handles label correlations, predicts multiple labels naturally',
      cons: 'More complex than single-label, label space can be huge',
      details: 'Binary Relevance trains C binary classifiers. Classifier Chains passes predictions as features. Neural nets use sigmoid per output. Label Powerset has 2^C classes.'
    },

    manyClassAlgo: {
      name: 'Large Number of Classes (>100)',
      algorithms: ['Hierarchical Softmax', 'Neural Networks with embedding layers', 'One-vs-all SVM', 'Extreme Multi-label classification (XML)', 'Nearest Class Mean', 'Siamese/Triplet Networks'],
      why: 'You have many classes (>100):\n\n• Hierarchical Softmax: O(log C) instead of O(C)\n• Embeddings: Maps to continuous space\n• One-vs-all: Efficient with sparse data\n• XML methods: Designed for millions of labels\n• Metric learning: Learns similarity metric\n• Reduces computational complexity',
      when: 'Use for fine-grained classification, entity recognition with large vocabularies',
      pros: 'Scales to thousands/millions of classes efficiently',
      cons: 'More complex implementation, may need special libraries',
      details: 'Hierarchical softmax organizes classes in tree. Embeddings use cosine similarity. Siamese networks learn distance metrics. XML uses tree-based indexing.'
    },

    kmeans: {
      name: 'K-Means and Variants',
      algorithms: ['K-Means', 'K-Means++', 'Mini-batch K-Means', 'K-Medoids (PAM)', 'Fuzzy C-Means'],
      why: 'You know the number of spherical clusters:\n\n• K-Means: Fast, simple, scales well\n• K-Means++: Better initialization than random\n• Mini-batch: Faster for large data\n• K-Medoids: Robust to outliers, uses actual points as centers\n• Fuzzy C-Means: Soft clustering, points belong to multiple clusters\n• All assume spherical, convex clusters',
      when: 'Use for customer segmentation, image compression, preprocessing',
      pros: 'Very fast, simple to implement, scales well, widely used',
      cons: 'Need to specify K, assumes spherical clusters, sensitive to initialization and outliers',
      details: 'Alternates between assigning points to nearest center and updating centers. K-Means++ picks initial centers with probability proportional to distance. Mini-batch uses random samples.'
    },

    dbscan: {
      name: 'Density-Based Clustering',
      algorithms: ['DBSCAN', 'OPTICS', 'HDBSCAN', 'Mean Shift', 'DENCLUE'],
      why: 'You have arbitrary shapes or unknown number of clusters:\n\n• DBSCAN: Finds clusters of varying shapes, detects outliers\n• OPTICS: Like DBSCAN but for varying densities\n• HDBSCAN: Hierarchical version, automatic parameter selection\n• Mean Shift: Finds modes of density, no K needed\n• Can discover non-convex clusters\n• Noise/outlier detection included',
      when: 'Use when clusters have arbitrary shapes or you don\'t know K',
      pros: 'No need to specify K, finds arbitrary shapes, robust to outliers',
      cons: 'Slower than K-Means, sensitive to parameters (eps, min_samples), struggles with varying densities',
      details: 'DBSCAN groups points with many neighbors within epsilon distance. OPTICS builds reachability plot. HDBSCAN builds cluster hierarchy. Mean Shift iteratively shifts points toward density modes.'
    },

    hierarchical: {
      name: 'Hierarchical Clustering',
      algorithms: ['Agglomerative Clustering (bottom-up)', 'Divisive Clustering (top-down)', 'BIRCH (Balanced Iterative Reducing)', 'Linkage methods (Single, Complete, Average, Ward)'],
      why: 'You need hierarchical structure or dendrogram:\n\n• Produces tree of clusters (dendrogram)\n• Can cut at any level for different K\n• Shows relationships between clusters\n• Ward linkage minimizes within-cluster variance\n• BIRCH efficient for large datasets\n• No need to specify K upfront',
      when: 'Use for taxonomy creation, when cluster hierarchy matters',
      pros: 'Produces dendrogram, no need to specify K, deterministic',
      cons: 'O(n²) or O(n³) time complexity, cannot undo merges/splits, sensitive to noise',
      details: 'Agglomerative starts with each point as cluster, merges closest pairs. Divisive starts with one cluster, recursively splits. Ward minimizes variance increase. BIRCH uses tree structure.'
    },

    gmm: {
      name: 'Gaussian Mixture Models',
      algorithms: ['Gaussian Mixture Models (GMM)', 'Bayesian Gaussian Mixture', 'Variational Inference for GMM'],
      why: 'You need probabilistic/soft clustering:\n\n• Models data as mixture of Gaussians\n• Each point has probability of belonging to each cluster\n• Can capture elliptical clusters\n• Provides uncertainty estimates\n• Can model overlapping clusters\n• Bayesian version automatically determines K',
      when: 'Use when clusters overlap or you need probability estimates',
      pros: 'Soft clustering, probabilistic, can model covariance, flexible cluster shapes',
      cons: 'Assumes Gaussian distributions, sensitive to initialization, can overfit',
      details: 'Uses Expectation-Maximization (EM) algorithm. E-step computes probabilities. M-step updates parameters. Bayesian version uses Dirichlet process prior.'
    },

    largeClustering: {
      name: 'Large-Scale Clustering',
      algorithms: ['Mini-batch K-Means', 'BIRCH', 'CLARA', 'Streaming clustering (StreamKM++)', 'Approximate methods'],
      why: 'You have very large datasets (millions of points):\n\n• Mini-batch K-Means: Uses random batches, much faster\n• BIRCH: Builds tree structure, single pass\n• CLARA: Samples and clusters subsets\n• Streaming: Processes data in one pass\n• Trade accuracy for speed and memory',
      when: 'Use when data is too large for regular clustering',
      pros: 'Scales to millions of points, memory efficient, fast',
      cons: 'May sacrifice some accuracy, approximate results',
      details: 'Mini-batch uses random samples each iteration. BIRCH uses CF-tree. CLARA samples multiple times and picks best. Streaming maintains summary statistics.'
    },

    pca: {
      name: 'Principal Component Analysis (PCA)',
      algorithms: ['PCA', 'Incremental PCA', 'Kernel PCA', 'Sparse PCA', 'Robust PCA'],
      why: 'You want linear dimensionality reduction:\n\n• Finds directions of maximum variance\n• Orthogonal components\n• Fast and efficient\n• Incremental PCA for large data\n• Kernel PCA for non-linear manifolds\n• Sparse PCA for interpretable components\n• Reduces noise and computational cost',
      when: 'Use for preprocessing, visualization, noise reduction',
      pros: 'Fast, interpretable, preserves variance, widely used',
      cons: 'Linear only, assumes Gaussian data, components may not be interpretable',
      details: 'Computes eigenvectors of covariance matrix. Projects data onto top k eigenvectors. Kernel PCA uses kernel trick. Incremental updates for streaming data.'
    },

    visualization: {
      name: 'Visualization (2D/3D)',
      algorithms: ['t-SNE', 'UMAP', 'PaCMAP', 'TriMAP', 'LargeVis', 'Isomap', 'Locally Linear Embedding (LLE)'],
      why: 'You want to visualize high-dimensional data:\n\n• t-SNE: Preserves local structure, reveals clusters\n• UMAP: Faster than t-SNE, preserves global structure better\n• PaCMAP: Balances local and global structure\n• All reduce to 2-3 dimensions for plotting\n• Reveals hidden patterns and clusters\n• Great for exploratory analysis',
      when: 'Use to understand data structure, present to stakeholders',
      pros: 'Beautiful visualizations, reveals clusters, intuitive',
      cons: 'Slow for large data, stochastic (different runs differ), cannot transform new points (t-SNE)',
      details: 't-SNE minimizes KL divergence between high-dim and low-dim probabilities. UMAP builds fuzzy topological representation. PaCMAP balances local/mid-range/global distances.'
    },

    lda: {
      name: 'Linear Discriminant Analysis',
      algorithms: ['Linear Discriminant Analysis (LDA)', 'Quadratic Discriminant Analysis', 'Regularized Discriminant Analysis'],
      why: 'You want supervised dimensionality reduction:\n\n• Uses class labels to find discriminative directions\n• Maximizes between-class variance\n• Minimizes within-class variance\n• Both classifier and dimensionality reduction\n• Can reduce to C-1 dimensions (C classes)\n• Often used before other classifiers',
      when: 'Use when you have labels and want discriminative features',
      pros: 'Supervised, finds discriminative directions, can classify',
      cons: 'Assumes Gaussian classes, linear boundaries, needs labels',
      details: 'Solves generalized eigenvalue problem with between and within-class scatter matrices. QDA allows different covariances per class. RDA adds regularization.'
    },

    sparseMethod: {
      name: 'Sparse/Feature Selection Methods',
      algorithms: ['L1-regularized methods (Lasso)', 'Elastic Net', 'Recursive Feature Elimination', 'Mutual Information', 'Chi-squared test', 'Feature importance from trees', 'SHAP values'],
      why: 'You have high-dimensional sparse data:\n\n• L1 regularization zeros out features\n• Removes irrelevant/redundant features\n• Improves interpretability\n• Reduces overfitting\n• Faster training and prediction\n• Lower memory usage',
      when: 'Use with high-dimensional data, when you need interpretability',
      pros: 'Reduces dimensions, improves interpretability, faster, prevents overfitting',
      cons: 'May remove useful features, requires tuning',
      details: 'Lasso adds L1 penalty that shrinks coefficients to zero. RFE recursively removes features. Mutual Information measures dependence. SHAP uses game theory.'
    },

    manifold: {
      name: 'Manifold Learning',
      algorithms: ['Isomap', 'Locally Linear Embedding (LLE)', 'Laplacian Eigenmaps', 'Autoencoders', 'Variational Autoencoders (VAE)'],
      why: 'You have non-linear manifolds:\n\n• Data lies on curved lower-dimensional manifold\n• Isomap preserves geodesic distances\n• LLE preserves local neighborhoods\n• Autoencoders learn non-linear encoding\n• VAE adds probabilistic structure\n• Captures intrinsic dimensionality',
      when: 'Use when linear methods fail, data has curved structure',
      pros: 'Handles non-linear manifolds, finds intrinsic dimensions',
      cons: 'Slow, sensitive to parameters, may not generalize to new points',
      details: 'Isomap uses shortest paths on nearest neighbor graph. LLE reconstructs each point from neighbors. Autoencoders use neural networks for encoding/decoding.'
    },

    textSeqClass: {
      name: 'Text Sequence Classification',
      algorithms: ['RNNs/LSTMs', 'GRUs', 'Bidirectional LSTMs', 'Transformers (BERT, RoBERTa)', 'GPT fine-tuned', 'DistilBERT', 'ELECTRA', 'DeBERTa'],
      why: 'You need to classify text sequences:\n\n• LSTMs/GRUs: Handle sequential dependencies\n• Bidirectional: See context from both directions\n• BERT: Pre-trained on massive text, understands context\n• GPT: Autoregressive, can do few-shot\n• DistilBERT: Faster, smaller BERT\n• State-of-the-art on most NLP tasks',
      when: 'Use for sentiment analysis, intent classification, question answering',
      pros: 'Understands context, handles variable length, state-of-the-art accuracy',
      cons: 'Requires GPUs, slow inference, large models',
      details: 'LSTM uses gates to control information flow. BERT uses bidirectional transformers with masked language modeling pre-training. GPT uses causal attention.'
    },

    textGeneration: {
      name: 'Text Generation',
      algorithms: ['GPT-2/3/4', 'LLaMA', 'Transformer-XL', 'XLNet', 'T5', 'BART', 'PaLM', 'Claude'],
      why: 'You want to generate text:\n\n• GPT: Autoregressive generation, coherent long text\n• LLaMA: Open source alternative\n• T5: Text-to-text framework\n• BART: Encoder-decoder for generation\n• These are large language models (LLMs)\n• Can do few-shot and zero-shot tasks',
      when: 'Use for content creation, dialogue, summarization',
      pros: 'Coherent text, creative, can follow instructions',
      cons: 'Very large, expensive to run, can hallucinate',
      details: 'GPT uses transformer decoder with causal attention. Trained to predict next token. T5 frames everything as text-to-text. BART denoises corrupted text.'
    },

    translation: {
      name: 'Machine Translation',
      algorithms: ['Transformer (attention is all you need)', 'Seq2Seq with attention', 'mBERT', 'mT5', 'M2M-100', 'NLLB'],
      why: 'You need to translate between languages:\n\n• Transformer: Revolutionized translation with attention\n• Seq2Seq: Encoder-decoder architecture\n• Multilingual models: Support many language pairs\n• M2M: Direct translation without English pivot\n• NLLB: Supports 200+ languages',
      when: 'Use for translating text between languages',
      pros: 'High quality, supports many languages, can handle rare words',
      cons: 'Requires parallel corpora, large models, domain-specific',
      details: 'Transformer uses encoder-decoder with multi-head self-attention and cross-attention. Trained on parallel sentences. Uses beam search for decoding.'
    },

    timeSeriesForecast: {
      name: 'Time Series Forecasting (Sequence)',
      algorithms: ['LSTM/GRU', 'Temporal Convolutional Networks', 'Transformers (Temporal Fusion Transformer)', 'N-BEATS', 'DeepAR', 'Prophet', 'WaveNet'],
      why: 'You need to forecast future values:\n\n• LSTM: Learns long-term dependencies\n• TCN: Uses dilated convolutions, parallelizable\n• Transformers: Attention over time\n• N-BEATS: Interpretable deep learning\n• DeepAR: Probabilistic, works across many series\n• WaveNet: Originally for audio, works for time series',
      when: 'Use for demand forecasting, stock prediction, weather',
      pros: 'Handles complex patterns, multi-step forecasting, probabilistic',
      cons: 'Needs lots of data, computationally expensive, black box',
      details: 'LSTM processes sequence step-by-step. TCN uses causal convolutions. Transformers use positional encoding. DeepAR models conditional distribution.'
    },

    speechRecog: {
      name: 'Speech Recognition',
      algorithms: ['Wav2Vec 2.0', 'Whisper', 'DeepSpeech', 'Listen, Attend and Spell (LAS)', 'Conformer', 'Streaming models'],
      why: 'You need to convert speech to text:\n\n• Wav2Vec: Self-supervised pre-training on raw audio\n• Whisper: Robust, multilingual, from OpenAI\n• Conformer: Combines convolution and attention\n• LAS: Attention-based encoder-decoder\n• Can handle various accents and noise',
      when: 'Use for transcription, voice commands, accessibility',
      pros: 'High accuracy, handles accents, end-to-end',
      cons: 'Requires lots of audio data, computationally expensive',
      details: 'Wav2Vec uses contrastive learning on masked audio. Whisper trained on 680K hours. Conformer uses convolution for local context, attention for global.'
    },

    semiSupervisedAlgo: {
      name: 'Semi-Supervised Learning',
      algorithms: ['Self-training', 'Co-training', 'Label Propagation', 'Pseudo-labeling', 'MixMatch', 'FixMatch', 'Meta Pseudo Labels', 'Ladder Networks'],
      why: 'You have mostly unlabeled data:\n\n• Self-training: Train on labeled, predict unlabeled, retrain\n• Label Propagation: Spreads labels through graph\n• MixMatch: Combines consistency regularization and pseudo-labeling\n• FixMatch: Uses weak and strong augmentation\n• Leverages structure in unlabeled data\n• Reduces labeling cost',
      when: 'Use when labeling is expensive but unlabeled data is abundant',
      pros: 'Uses unlabeled data, improves with more data, reduces labeling cost',
      cons: 'Can propagate errors, sensitive to initial labeled set',
      details: 'Self-training iteratively adds confident predictions. Label Propagation builds k-NN graph. MixMatch does consistency regularization with mixup.'
    },

    supervisedAnomaly: {
      name: 'Supervised Anomaly Detection',
      algorithms: ['Classification algorithms with class imbalance techniques', 'One-Class SVM', 'SVDD (Support Vector Data Description)', 'Isolation Forest (semi-supervised)', 'Autoencoders trained on normal data'],
      why: 'You have labeled anomalies:\n\n• Treat as imbalanced classification\n• One-Class SVM: Learns boundary around normal\n• Isolation Forest: Isolates anomalies with fewer splits\n• Autoencoders: High reconstruction error on anomalies\n• Use techniques like SMOTE, class weights',
      when: 'Use for fraud detection, defect detection with labeled examples',
      pros: 'Can learn specific anomaly types, higher accuracy',
      cons: 'Requires anomaly labels, may not generalize to new anomaly types',
      details: 'One-Class SVM finds minimum enclosing hypersphere. Isolation Forest uses random trees. Autoencoders learn to reconstruct, fail on anomalies.'
    },

    unsupervisedAnomaly: {
      name: 'Unsupervised Anomaly Detection',
      algorithms: ['Isolation Forest', 'Local Outlier Factor (LOF)', 'One-Class SVM', 'Autoencoders', 'DBSCAN', 'Gaussian Mixture Models', 'Robust Covariance'],
      why: 'You have no labeled anomalies:\n\n• Isolation Forest: Fast, works well in practice\n• LOF: Based on local density\n• One-Class SVM: Learns normal data boundary\n• Autoencoders: Reconstruction error\n• Assumes anomalies are rare and different\n• No labels needed',
      when: 'Use when you cannot label anomalies or they are unknown',
      pros: 'No labels needed, can find novel anomalies',
      cons: 'May have false positives, hard to evaluate without labels',
      details: 'Isolation Forest randomly partitions, anomalies isolated quickly. LOF compares density to neighbors. Autoencoders fail to reconstruct unusual patterns.'
    },

    onlineAnomaly: {
      name: 'Online/Streaming Anomaly Detection',
      algorithms: ['Online learning algorithms', 'Exponentially Weighted Moving Average (EWMA)', 'Streaming quantiles', 'Incremental PCA', 'Online clustering updates', 'ARIMA residuals', 'Streaming isolation forest'],
      why: 'You have real-time streaming data:\n\n• Must process data in one pass\n• Update model incrementally\n• EWMA: Simple, tracks moving statistics\n• Streaming quantiles: Detect distribution changes\n• Immediate detection required\n• Memory constraints',
      when: 'Use for real-time monitoring, IoT sensors, network traffic',
      pros: 'Real-time, memory efficient, adapts to drift',
      cons: 'May miss complex patterns, sensitive to parameters',
      details: 'EWMA updates statistics with decay. Streaming quantiles use online algorithms. Incremental PCA updates singular vectors. Compare new points to learned distribution.'
    },

    associationAlgo: {
      name: 'Association Rule Mining',
      algorithms: ['Apriori', 'FP-Growth', 'ECLAT', 'Frequent Pattern Mining'],
      why: 'You want to find frequently co-occurring items:\n\n• Apriori: Classic algorithm, uses candidate generation\n• FP-Growth: Faster, uses tree structure\n• ECLAT: Depth-first search\n• Finds rules like "if A and B then C"\n• Market basket analysis\n• Measures support, confidence, lift',
      when: 'Use for market basket analysis, recommendation systems',
      pros: 'Finds interpretable rules, handles large itemsets',
      cons: 'Exponential search space, many redundant rules, discrete data only',
      details: 'Apriori generates candidates level-wise. FP-Growth builds frequent pattern tree. ECLAT uses vertical data format. Filter by minimum support and confidence.'
    },

    densityAlgo: {
      name: 'Density Estimation',
      algorithms: ['Kernel Density Estimation (KDE)', 'Gaussian Mixture Models', 'Histograms', 'Variational Autoencoders', 'Normalizing Flows', 'Energy-Based Models'],
      why: 'You want to estimate probability density:\n\n• KDE: Non-parametric, smooth estimates\n• GMM: Parametric mixture model\n• VAE: Deep generative model\n• Normalizing Flows: Exact likelihood\n• Useful for sampling, anomaly detection\n• Understanding data distribution',
      when: 'Use when you need probability estimates, sampling, or generation',
      pros: 'Models full distribution, can sample, detects outliers',
      cons: 'Curse of dimensionality, requires careful tuning',
      details: 'KDE sums kernels around points. GMM fits mixture of Gaussians. VAE learns latent distribution. Normalizing Flows use invertible transformations.'
    },

    generativeType: {
      name: 'Generative Models',
      algorithms: ['Generative Adversarial Networks (GANs)', 'Variational Autoencoders (VAE)', 'Diffusion Models', 'Normalizing Flows', 'PixelCNN', 'Vector Quantized VAE (VQ-VAE)', 'StyleGAN', 'Stable Diffusion'],
      why: 'You want to generate new samples:\n\n• GANs: High quality images, adversarial training\n• VAE: Probabilistic, smooth latent space\n• Diffusion: State-of-the-art image generation\n• Normalizing Flows: Exact likelihood\n• Generate realistic images, text, audio\n• Data augmentation, creative applications',
      when: 'Use for data augmentation, creative content, synthetic data',
      pros: 'Generate realistic samples, learn data distribution, creative applications',
      cons: 'Hard to train (GANs), computationally expensive, can memorize training data',
      details: 'GANs train generator and discriminator adversarially. VAE uses variational inference. Diffusion gradually adds then removes noise. Flows use invertible networks.'
    },

    rankingAlgo: {
      name: 'Learning to Rank',
      algorithms: ['LambdaMART', 'RankNet', 'ListNet', 'XGBoost for ranking', 'Neural ranking models', 'BERT for ranking'],
      why: 'You need to rank items:\n\n• LambdaMART: Gradient boosting for ranking\n• RankNet: Neural network with pairwise loss\n• ListNet: Learns to rank entire lists\n• Used in search engines\n• Optimizes ranking metrics (NDCG, MAP)\n• Can incorporate multiple features',
      when: 'Use for search engines, recommendation ranking, ad placement',
      pros: 'Optimizes ranking metrics directly, handles position bias',
      cons: 'Requires relevance labels, complex training',
      details: 'LambdaMART uses gradients on ranking metrics. RankNet uses pairwise preferences. ListNet uses probability distributions over permutations.'
    },

    recommendationType: {
      name: 'Recommendation Systems',
      algorithms: ['Collaborative Filtering (User-based, Item-based)', 'Matrix Factorization (SVD, NMF)', 'Deep Learning (Neural Collaborative Filtering)', 'Content-based filtering', 'Hybrid methods', 'Factorization Machines', 'Two-tower models', 'Graph-based (Node2Vec)'],
      why: 'You want to recommend items to users:\n\n• Collaborative Filtering: Uses user-item interactions\n• Matrix Factorization: Learns latent factors\n• Deep Learning: Learns complex patterns\n• Content-based: Uses item features\n• Hybrid: Combines multiple approaches\n• Cold start solutions for new users/items',
      when: 'Use for product recommendations, content recommendations',
      pros: 'Personalized, improves engagement, learns preferences',
      cons: 'Cold start problem, sparsity, scalability challenges',
      details: 'CF finds similar users/items. MF decomposes user-item matrix. NCF uses neural networks. Two-tower separately encodes users and items.'
    }
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
    setShowExplanation(false);
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
                  const hasDetailPage = ['elasticnet', 'bayesianlinearregression', 'robustregression', 'huberregression', 'ransac', 'theilsenestimator', 'lassoregressionl1', 'lasso', 'ridgeregressionl2', 'ridge'].includes(algoKey);
                  
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