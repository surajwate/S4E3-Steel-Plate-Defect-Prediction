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
        X['Y_Range'] = X['Y_Maximum'] - X['Y_Minimum']
        X['Area_Perimeter_Ratio'] = X['Pixels_Areas'] / (X['X_Perimeter'] + X['Y_Perimeter'])
        X['Luminosity_Range'] = X['Maximum_of_Luminosity'] - X['Minimum_of_Luminosity']
        X['Volume'] = X['X_Range'] * X['Y_Range'] * X['Steel_Plate_Thickness']
        X['Thickness_Deviation'] = X['Steel_Plate_Thickness'] - self.mean_thickness
        return X