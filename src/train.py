import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import roc_auc_score

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from feature_engineering import FeatureEngineer

import os
import argparse
import joblib

import config
import model_dispatcher

import time
import logging

# Set up logging
logging.basicConfig(
    filename=config.LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s"
)

def run(fold, model):
    # Import the data
    df = pd.read_csv(config.TRAINING_FILE)

    # Split the data into training and testing
    train = df[df.kfold != fold].reset_index(drop=True)
    test = df[df.kfold == fold].reset_index(drop=True)

    # Split the data into features and target
    target_features = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']

    X_train = train.drop(['id', 'kfold', 'Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults'], axis=1)
    X_test = test.drop(['id', 'kfold', 'Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults'], axis=1)

    y_train = train[target_features].values
    y_test = test[target_features].values

    # Define features
    features = X_train.columns

    # Create a column transformer for one-hot encoding and standard scaling
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), features)
        ]
    )

    # Create a pipeline with the preprocessor and the model
    pipeline = Pipeline(steps=[
        ('feature_engineer', FeatureEngineer()),
        ('preprocessor', preprocessor),
        ('model', model_dispatcher.models[model])
    ])


    try:
        start = time.time()

        # logging.info(f"Fold={fold}, Model={model}")

        # Fit the model
        pipeline.fit(X_train, y_train)

        # make probability predictions
        preds = pipeline.predict_proba(X_test)

        end = time.time()
        time_taken = end - start

        # Calculate the AUC score for each of the 7 classes
        auc_scores = []
        for i, col in enumerate(target_features):
            # For Random Forest or any tree-based model, `predict_proba` returns a list of arrays
            # Extract the probabilities of the positive class (index=1) for each class
            if isinstance(preds, list):
                auc = roc_auc_score(y_test[:, i], preds[i][:, 1])
            else:
                auc = roc_auc_score(y_test[:, i], preds[:, i])
            auc_scores.append(auc)
            print(f"{col} - AUC={auc:.4f}")

        # Final average AUC score
        final_score = np.mean(auc_scores)
        print(f"Final Average AUC={final_score:.4f}")

        logging.info(f"Fold={fold}, Final Average AUC={final_score:.4f}, Time Taken={time_taken:.2f}sec")

        # Save the model
        joblib.dump(pipeline, os.path.join(config.MODEL_OUTPUT, f"model_{fold}.bin"))
    except Exception as e:
        logging.exception(f"Error occurred for Fold={fold}, Model={model}: {str(e)}")
    

if __name__ == '__main__':
    # Initialize the ArgumentParser class of argparse
    parser = argparse.ArgumentParser()

    # Add the arguments to the parser
    parser.add_argument("--fold", type=int)
    parser.add_argument("--model", type=str)

    # Read the arguments from the command line
    args = parser.parse_args()

    # Run the fold specified by the command line arguments
    run(fold=args.fold, model=args.model)

