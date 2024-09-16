from sklearn.base import BaseEstimator, TransformerMixin

# Custom transformer for creating new features
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # Calculate any statistics based on training data
        self.mean_thickness = X['Steel_Plate_Thickness'].mean()
        return self

    def transform(self, X):
        X = X.copy()
        X['X_Range'] = X['X_Maximum'] - X['X_Minimum']
        X['Thickness_Deviation'] = X['Steel_Plate_Thickness'] - self.mean_thickness
        return X