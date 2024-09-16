# Steel Plate Defect Prediction

This project aims to build a multi-label classification model to identify the type of faults in steel plates, using a synthetically generated dataset from the UCI Repository. The task is to predict the presence of seven different types of faults in steel plates.

- **Kaggle Dataset**: [Playground Series: Season 4, Episode 3](https://www.kaggle.com/competitions/playground-series-s4e3/data)
- **UCI Repository Dataset**: [Steel Plates Faults](https://archive.ics.uci.edu/dataset/198/steel+plates+faults)

## Project Structure

The project is organized as follows:

```plaintext
Steel-Plate-Defect-Prediction/
│
├── docs/                   # Documentation (if any)
├── input/                  # Raw input data files
├── logs/                   # Logging information for models
├── models/                 # Saved model files after training
├── notebooks/              # Jupyter Notebooks used for EDA and experiments
│   ├── eda.ipynb           # Exploratory Data Analysis notebook
│   ├── submission_code.ipynb # Code used for final submissions
├── output/                 # Output files (such as predictions and submissions)
│   └── submission.csv      # Final submission file
├── src/                    # Source code files for the project
│   ├── config.py           # Configuration file with paths and settings
│   ├── create_fold.py      # Code for creating K-fold splits
│   ├── feature_engineering.py # Feature engineering classes and functions
│   ├── final_model.py      # Code to train the final model
│   ├── model_dispatcher.py # Dictionary mapping model names to implementations
│   ├── train.py            # Main training file with model pipelines
├── run.ps1                 # Script to run the training pipeline on Windows
├── run.py                  # Script to run the training pipeline on Linux/MacOS
├── README.md               # Project documentation (this file)
└── .gitignore              # Git ignore file
```

## How to Use

### 1. Clone the repository

```bash
git clone https://github.com/surajwate/S4E3-Steel-Plate-Defect-Prediction.git
cd S4E3-Steel-Plate-Defect-Prediction
```

### 2. Install dependencies

Ensure that you have Python 3.7+ installed. Then, install the necessary dependencies using:

```bash
pip install -r requirements.txt
```

### 3. Data Preparation

Make sure to download the required datasets from the Kaggle competition or UCI repository and place them in the `input/` directory.

### 4. Create K-Folds

To create K-fold splits for cross-validation, run the `create_fold.py` script:

```bash
python src/create_fold.py
```

This will create new columns in the dataset to store the fold information.

### 5. Train Models

You can train the models by specifying the model name and folds. Run the following command to train all models in cross-validation mode:

- For Windows:

```bash
python run.ps1
```

- For Linux or macOS:

```bash
python run.py
```

#### Train individual model

```bash
python main.py --model <model_name>
```

Replace `<model_name>` with one of the following models: `logistic_regression`, `random_forest`, `xgboost`, `lightgbm`, `catboost`, etc.

### 6. Feature Engineering

Feature engineering is handled in `feature_engineering.py`. The following new features were added but did not improve performance significantly:

```python
X['X_Range'] = X['X_Maximum'] - X['X_Minimum']
X['Y_Range'] = X['Y_Maximum'] - X['Y_Minimum']
X['Area_Perimeter_Ratio'] = X['Pixels_Areas'] / (X['X_Perimeter'] + X['Y_Perimeter'])
X['Luminosity_Range'] = X['Maximum_of_Luminosity'] - X['Minimum_of_Luminosity']
X['Volume'] = X['X_Range'] * X['Y_Range'] * X['Steel_Plate_Thickness']
X['Thickness_Deviation'] = X['Steel_Plate_Thickness'] - mean_thickness
```

### 7. Model Evaluation

The model performance is evaluated using the Area Under the ROC Curve (AUC) for each of the seven fault types, and then averaged across all faults. After training, the models are saved in the `models/` directory.

The results of various models can be seen below:

| Model               | Fold 0 AUC | Fold 1 AUC | Fold 2 AUC | Fold 3 AUC | Fold 4 AUC | Avg AUC | Time (sec) |
|---------------------|------------|------------|------------|------------|------------|---------|------------|
| logistic_regression  | 0.8652     | 0.8564     | 0.8633     | 0.8587     | 0.8553     | 0.8598  | 0.32       |
| random_forest        | 0.8790     | 0.8747     | 0.8732     | 0.8726     | 0.8699     | 0.8739  | 7.75       |
| xgboost              | 0.8766     | 0.8721     | 0.8725     | 0.8719     | 0.8722     | 0.8731  | 1.18       |
| lightgbm             | 0.8851     | 0.8822     | 0.8802     | 0.8811     | 0.8801     | 0.8817  | 1.04       |
| catboost             | 0.8888     | 0.8832     | 0.8838     | 0.8854     | 0.8831     | 0.8849  | 60.02      |

## Final Model Submission

The best model, `CatBoost`, was selected for submission after training on the entire training set. The final score on the public leaderboard was **0.88169**, ranking between 950-951.

## Key Learnings

1. **Handling Multi-label Classification**: Certain models don't natively support multi-label classification. Wrapping models using `OneVsRestClassifier` helped in this regard.
2. **Predict Method**: Each model's `predict` and `predict_proba` methods can return results in different formats (list vs arrays), requiring special handling.
3. **Feature Engineering**: While feature engineering was performed, it did not significantly improve model performance, possibly due to pre-engineered features in the dataset.

## Useful Links

- **Notebook**: [Kaggle Notebook for S4E3](https://www.kaggle.com/code/surajwate/s4e3-streel-plate-defect)
- **Blog Post**: [Steel Plate Defect Prediction](https://surajwate.com/blog/steel-plate-defect-prediction/)

---

This README file outlines the project structure, how to use the project, key details of the models and results, as well as links to the notebook and blog for further insights.
