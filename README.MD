# Spaceship Titanic Prediction

This project aims to predict whether a passenger aboard the Spaceship Titanic was transported to another dimension based on their demographics, travel status, and onboard activity. The solution follows a complete machine learning pipeline including data cleaning, feature engineering, model optimization, and interpretability using permutation importance.

## Dataset

- **Source**: [Kaggle – Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic)
- **Files Used**:
  - `train.csv`: Training dataset with labels
  - `test.csv`: Unlabeled dataset for submission
- **Size**:
  - ~8,700 records in training
  - ~4,300 records in test

## Objective

Predict the `Transported` status (boolean: `True` or `False`) for passengers based on spaceflight features. The project also investigates which engineered and original features influence predictions the most and evaluates multiple model selection strategies.

## Methodology

### Exploratory Data Analysis (EDA)
- Examined distributions of numerical and categorical features.
- Plotted amenity spending vs. target variable.
- Computed Phi-K correlation to uncover nonlinear associations.
- Identified skewed distributions and class balance.

### Statistical Inference
- Applied Chi-Square tests and confidence intervals to evaluate significance of:
  - `CryoSleep`, `HomePlanet`, `Deck`, `Side`, `SpentAnything`
- Strong statistical association found with `CryoSleep` and `HomePlanet`.

### Feature Engineering
- Created features from compound columns:
  - `GroupID`, `GroupSize`, `GroupMember` from `PassengerId`
  - `Deck`, `CabinNum`, `Side` from `Cabin`
  - `FirstName`, `LastName` from `Name`
  - `SpentAnything` from sum of amenity spend columns

### Preprocessing Pipelines
- Used `ColumnTransformer` with:
  - Mean imputation + `StandardScaler` for numerical features
  - Mode imputation + `OneHotEncoder` for categorical features

### Modeling & Tuning
- Models trained and tuned using:
  - **Baseline**: Logistic Regression
  - **RandomizedSearchCV** for hyperparameter tuning of:
    - Logistic Regression, SVM, Random Forest, CatBoost, XGBoost
  - **Optuna**: Bayesian optimization with 20 trials per model
  - **FLAML**: AutoML strategy with 5-minute time budget

### Ensemble Learning
- Built soft-voting ensembles from top 3 models in each tuning strategy.
- Tuned classification threshold (0.1–0.9) for each ensemble based on validation accuracy.

### Evaluation Metrics
- Accuracy, F1 Score, Precision, Recall
- ROC AUC and PR AUC
- Confusion matrix and classification report

### Model Interpretation
- Applied **permutation feature importance** to:
  - Random Search Voting Ensemble
  - Optuna Voting Ensemble
  - FLAML best model (typically CatBoost)
- Features like `VRDeck`, `Spa`, and `FoodCourt` were consistently important.
- Features like `GroupMember`, `GroupSize`, and `HomePlanet` showed low or negative importance in ensembles.

## Final Results

- **Best model**: Voting Ensemble (Optuna)
- **Validation Accuracy**: ~81.1%
- **Final Submission Score** (Kaggle Private LB): **0.80126**
- **Threshold used**: Varies by model (optimized, e.g., 0.38 for Random, 0.41 for Optuna)

## Docker Support

A `Dockerfile` is included for reproducibility and GPU-enabled environments (CUDA 12.0 + cuDNN 8). It installs all dependencies including:

- Python 3.10, JupyterLab
- XGBoost, CatBoost, FLAML, Optuna
- scikit-learn, SHAP, phik, seaborn, statsmodels

### To Build and Run:

```bash
docker build -t spaceship-titanic .
docker run --gpus all -p 8888:8888 -v $PWD:/workspace spaceship-titanic
```

Then open JupyterLab in your browser at `http://localhost:8888`.

## How to Run

1. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

2. **Run analysis**:

   ```bash
   jupyter notebook
   ```

   Open and execute all cells in the notebook (`spaceship-titanic.ipynb` or similar).

3. **Export submission**:

   The final cell creates a submission CSV based on the best model and saves it to disk.

## Project Structure

```
spaceship-titanic/
├── spaceship-titanic.ipynb            # Main notebook  
├── train.csv                          # Training data  
├── test.csv                           # Test data  
├── submission_final_*.csv             # Final model submission  
├── utils.py                           # Utility functions and plotting  
├── requirements.txt                   # Python dependencies  
├── Dockerfile                         # GPU-ready container for reproducible builds  
└── README.md                          # Project documentation
```

## License

This project is licensed under the MIT License.

## Acknowledgments

- Competition: [Spaceship Titanic on Kaggle](https://www.kaggle.com/competitions/spaceship-titanic)