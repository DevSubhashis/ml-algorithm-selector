# 🤖 ML Algorithm Selector

> An interactive web application that helps you choose the right machine learning algorithm for your project

[![React](https://img.shields.io/badge/React-18.0+-blue.svg)](https://reactjs.org/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind-3.0+-38B2AC.svg)](https://tailwindcss.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📖 Overview

ML Algorithm Selector is a comprehensive, interactive decision-tree based tool that guides you through selecting the most appropriate machine learning algorithm for your specific use case. Whether you're a student learning ML, a data scientist exploring options, or an engineer building production systems, this tool provides expert guidance with detailed explanations.

### ✨ Key Features

- 🎯 **Smart Decision Tree** - Answer targeted questions about your data and requirements
- 📚 **100+ Algorithms Covered** - From classical methods to state-of-the-art deep learning
- 💡 **Detailed Explanations** - Understand *why* each algorithm is recommended
- ⚖️ **Honest Trade-offs** - Learn the pros and cons of each approach
- 🎨 **Beautiful UI** - Modern, responsive design with smooth animations
- 🔄 **Step-by-step Navigation** - Go back and forth through your decisions
- 📱 **Fully Responsive** - Works on desktop, tablet, and mobile

## 🚀 Quick Start

### Prerequisites

- Node.js (v14 or higher)
- npm or yarn

### Installation

```bash
# Clone the repository
git clone https://github.com/DevSubhashis/ml-algorithm-selector.git

# Navigate to project directory
cd ml-algorithm-selector

# Install dependencies
npm install

# Start the development server
npm start
```

The application will open at `http://localhost:3000`

### Build for Production

```bash
npm run build
```

## 📊 Algorithm Categories Covered

### Supervised Learning
- **Regression**: Linear, Polynomial, Ridge, Lasso, Elastic Net, SVR, Random Forest, Gradient Boosting, Neural Networks
- **Classification**: Logistic Regression, SVM, Decision Trees, Random Forest, XGBoost, LightGBM, CatBoost, Neural Networks
- **Deep Learning**: CNNs, ResNet, EfficientNet, Vision Transformers

### Unsupervised Learning
- **Clustering**: K-Means, DBSCAN, Hierarchical, GMM, HDBSCAN, Mean Shift
- **Dimensionality Reduction**: PCA, t-SNE, UMAP, LDA, Autoencoders, Manifold Learning
- **Anomaly Detection**: Isolation Forest, LOF, One-Class SVM, Autoencoders
- **Association Rules**: Apriori, FP-Growth

### Sequence Models
- **NLP**: BERT, GPT, RoBERTa, T5, Transformers, LSTMs, GRUs
- **Time Series**: ARIMA, Prophet, LSTM, Temporal Fusion Transformer, N-BEATS
- **Speech**: Wav2Vec, Whisper, DeepSpeech

### Specialized
- **Semi-Supervised**: Label Propagation, MixMatch, FixMatch, Self-Training
- **Recommendation Systems**: Collaborative Filtering, Matrix Factorization, Neural CF
- **Ranking**: LambdaMART, RankNet, BERT for Ranking
- **Generative Models**: GANs, VAE, Diffusion Models, Normalizing Flows

## 🎮 How to Use

1. **Start**: Click on the application to begin
2. **Answer Questions**: Select options that best describe your:
   - Data characteristics (labeled/unlabeled, size, type)
   - Problem type (classification, regression, clustering, etc.)
   - Requirements (interpretability, speed, accuracy)
3. **Get Recommendations**: Receive tailored algorithm suggestions with:
   - Detailed explanation of why it fits your needs
   - List of specific algorithms to consider
   - Advantages and limitations
   - When to use and avoid
   - Technical implementation details
4. **Explore Further**: Go back to try different paths or start over

## 📸 Screenshots

### Decision Flow
```
Do you have labeled data?
  ↓
  Yes → What type of output?
    ↓
    Categorical → Classification
      ↓
      Binary/Multi-class → Data characteristics → Algorithm Recommendation
```

### Example Recommendation Screen
- Algorithm family name
- List of specific algorithms
- Why this was selected
- Pros and cons
- When to use
- Technical details

## 🛠️ Technology Stack

- **Frontend Framework**: React 18+
- **Styling**: Tailwind CSS 3+
- **Icons**: Lucide React
- **State Management**: React Hooks (useState)

## 📁 Project Structure

```
ml-algorithm-selector/
├── src/
│   ├── components/
│   │   └── MLAlgorithmSelector.jsx    # Main component
│   ├── App.js
│   └── index.js
├── public/
│   └── index.html
├── package.json
└── README.md
```

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. **Commit your changes**
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. **Push to the branch**
   ```bash
   git push origin feature/AmazingFeature
   ```
5. **Open a Pull Request**

### Ideas for Contribution
- Add more algorithms (Reinforcement Learning, Meta-Learning, etc.)
- Add algorithm comparison features
- Include code examples for each algorithm
- Add dataset size estimator
- Multi-language support
- Export recommendations as PDF
- Add algorithm performance benchmarks
- Integration with scikit-learn/TensorFlow documentation

## 📝 Algorithm Coverage Checklist

- [x] Classical ML (Linear Models, Trees, SVMs)
- [x] Ensemble Methods (Random Forest, Boosting)
- [x] Deep Learning (CNNs, RNNs, Transformers)
- [x] Clustering Algorithms
- [x] Dimensionality Reduction
- [x] Time Series Forecasting
- [x] NLP Models
- [x] Computer Vision Models
- [x] Anomaly Detection
- [x] Recommendation Systems
- [ ] Reinforcement Learning
- [ ] Meta-Learning
- [ ] Graph Neural Networks
- [ ] Federated Learning

## 🎓 Educational Use

This tool is perfect for:
- **ML Courses**: Use as a teaching aid
- **Workshops**: Guide participants through algorithm selection
- **Self-Learning**: Understand the ML algorithm landscape
- **Job Interviews**: Quick review of algorithm trade-offs

## 📚 Resources

Recommended resources to learn more about the algorithms:

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Papers with Code](https://paperswithcode.com/)
- [Machine Learning Mastery](https://machinelearningmastery.com/)
- [Fast.ai](https://www.fast.ai/)

## 🐛 Bug Reports

Found a bug? Please open an issue with:
- Description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Screenshots (if applicable)

## 💬 FAQ

**Q: Does this tell me the exact algorithm to use?**  
A: It provides recommendations based on common best practices. Always validate with your specific data.

**Q: Can I use this for production systems?**  
A: This is a guide. For production, conduct thorough experiments and benchmarks.

**Q: Are deep learning models always better?**  
A: No! Simple models often outperform complex ones with limited data. The tool helps you understand these trade-offs.

**Q: How accurate are the recommendations?**  
A: Based on industry best practices and academic consensus. However, always experiment with your specific data.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- Your Name - *Initial work* - [DevSubhashis](https://github.com/DevSubhashis)

## 🌟 Acknowledgments

- Inspired by scikit-learn's algorithm cheat sheet
- Built with knowledge from ML research papers and industry experience
- Thanks to the open-source ML community

## 📞 Contact

- GitHub: [@DevSubhashis](https://github.com/DevSubhashis)
- Email: jobshelper4u@gmail.com
- LinkedIn: [DevSubhashis](https://www.linkedin.com/in/subhashis-routh-1a3a71b5)

---

**Star ⭐ this repository if you find it helpful!**

Made with ❤️ and ☕ by [Subhahis Routh]