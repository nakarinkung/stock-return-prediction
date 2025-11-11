# Stock Return Prediction using Machine Learning  
### CF969-7-SP | Machine Learning for Finance – Assignment
---

## 📘 Overview  

This project explores the application of **supervised machine learning models** to predict **daily stock returns** using historical data and **technical indicators**.  
By leveraging models like **Linear Regression**, **Support Vector Machines (SVM)**, **Random Forests**, and **Neural Networks**, the study compares their predictive performance and discusses their implications in financial forecasting.

---

## 🧩 Objectives  

- Predict short-term **stock returns** using technical and market indicators.  
- Compare linear and nonlinear models to evaluate predictive accuracy.  
- Assess model robustness via **time-series cross-validation**.  
- Identify which model generalises best across multiple assets.

---

## 💾 Python libraries

pandas, numpy, matplotlib, seaborn, scikit-learn, tensorflow, statsmodels, yfinance

---


## 💾 Data Collection & Features  

- **Data Source:** [Yahoo Finance](https://pypi.org/project/yfinance/) (`yfinance` library)  
- **Time Period:** 5 years of daily data  
- **Stocks:**  
  - AAPL — Apple Inc. *(Technology)*  
  - AMZN — Amazon.com, Inc. *(E-commerce)*  
  - JPM — JPMorgan Chase & Co. *(Financials)*  
  - PG — Procter & Gamble Co. *(Consumer Staples)*  
  - UNH — UnitedHealth Group Inc. *(Healthcare)*  

### 🔧 Technical Indicators Used  
| Category | Indicator | Description |
|-----------|------------|-------------|
| Trend | SMA-10, SMA-50 | Moving averages for short & long-term trend detection |
| Momentum | RSI (14) | Measures overbought/oversold conditions |
| Volatility | ATR, Bollinger Bands | Captures market volatility |
| Momentum | MACD | Detects trend reversals |
| Market | S&P 500 Returns | Captures market-wide movements |

Data preprocessing includes **feature scaling** (StandardScaler) and **chronological train-test split (80/20)** to maintain temporal integrity.

---

## 🧠 Models Implemented  

### 1️⃣ Linear Regression  
Baseline model assuming linear relationships between predictors and target returns.  
- Implementation: `sklearn.linear_model.LinearRegression` and `statsmodels.OLS`  
- Significance testing via p-values for feature relevance  

### 2️⃣ Support Vector Regression (SVR)  
Captures non-linear relationships using kernel methods.  
- Kernels: Linear, Polynomial, RBF  
- Hyperparameter tuning: GridSearchCV (5-fold CV)  
- Key hyperparameters: `C`, `epsilon`, `gamma`, `degree`  

### 3️⃣ Random Forest Regressor  
An ensemble of decision trees capturing feature interactions.  
- Implementation: `sklearn.ensemble.RandomForestRegressor`  
- Feature importance extracted for interpretability  

### 4️⃣ Neural Network (Feedforward)  
A dense multi-layer model implemented with TensorFlow/Keras.  
- Architecture: 3 hidden layers (64–32–16 neurons)  
- Activation: ReLU (hidden), Linear (output)  
- Regularisation: Dropout (0.2) + L2 (0.001)  
- Optimizer: Adam | Loss: MSE | Early stopping (patience=10)  

---

## ⚙️ Model Evaluation  

Evaluation metrics:
- **MAE (Mean Absolute Error)** — average prediction error magnitude  
- **RMSE (Root Mean Squared Error)** — penalises large deviations  
- **R² (Coefficient of Determination)** — proportion of variance explained  

Time-series cross-validation (`TimeSeriesSplit`) ensures realistic, forward-looking evaluation.

---

## 📊 Results Summary  

| Stock | Best Model | Lowest MSE | Lowest MAE | Highest R² |
|--------|-------------|-------------|-------------|-------------|
| **AAPL** | SVM / NN | 0.000302 | 0.01311 | -0.0599 |
| **AMZN** | SVM | 0.000498 | 0.01607 | -0.0057 |
| **JPM** | SVM | 0.000234 | 0.01105 | -0.0097 |
| **PG** | Linear | 0.000121 | 0.00814 | -0.0628 |
| **UNH** | SVM | 0.000221 | 0.01059 | -0.0038 |

🧩 **Key Takeaways**
- **SVM** outperforms across most stocks — particularly for non-linear, volatile series.  
- **Linear Regression** performs best for more stable stocks (e.g., PG).  
- **Neural Networks** can reduce MAE but require careful tuning and larger datasets.  
- **Random Forests** show balanced but not leading performance.  

---


## 🔍 Insights  

- Market returns and volatility indicators (ATR, SMA_50) play major roles.  
- Non-linear models (SVM, NN) adapt better to complex patterns.  
- Linear models remain useful for interpretability and stable assets.  
- Incorporating macroeconomic or sentiment data could improve prediction accuracy.

---
