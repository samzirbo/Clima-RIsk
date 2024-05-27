from sklearn.preprocessing import LabelEncoder
from sklearn.base import TransformerMixin
import pandas as pd
import holidays
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier, XGBRegressor
import holidays
from sklearn.base import TransformerMixin
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import classification_report, recall_score, precision_score, log_loss, roc_auc_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import numpy as np
from imblearn.under_sampling import RandomUnderSampler

# Building Blocks
class BasicTransformations(TransformerMixin):

    def fit(self, X, y=None):
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(X['Sex'])
        return self

    def transform(self, df):
        X = df.copy()
        X['Sex'] = self.label_encoder.transform(X['Sex'])
        return X
    
class DateTransformer(TransformerMixin):
    def __init__(self, day='weekend', month=True, holiday=True):
        self.day = day
        self.month = month
        self.holiday = holiday

    def fit(self, X, y=None):
        return self

    def transform(self, df):
        X = df.copy()
        X['Date'] = pd.to_datetime(X['Date']).dt.date

        if self.day == 'dayofweek':
            X['Day'] = X['Date'].apply(lambda x: x.weekday())
        elif self.day == 'weekend':
            X['Weekend'] = X['Date'].apply(lambda x: 1 if x.weekday() >= 4 else 0)
        if self.month:
            X['Month'] = X['Date'].apply(lambda x: x.month)
        if self.holiday:
            min_year = X['Date'].min().year
            max_year = X['Date'].max().year
            holiday = list(holidays.Romania(years=range(min_year, max_year + 1)).keys())
            holiday += [date + pd.Timedelta(days=2) for date in holiday] + [date - pd.Timedelta(days=2) for date in holiday]
            X['Holiday'] = X['Date'].apply(lambda x: 1 if x in holiday else 0)

        X.drop(columns=['Date'], inplace=True)
        return X
    
    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self
    
class CityTransformer(TransformerMixin):
    def __init__(self, encoding='ohe', population=True, coordinates=True):
        self.encoding = encoding
        self.population = population
        self.coordinates = coordinates
        self.populations = {
            'Cluj Napoca': 322_108,
            'Timisoara': 333_613,
            'Iasi': 357_192,
            'Constanta': 319_168,
            'Bucuresti': 2_103_346,
        }
        self.city_coordinates = {
            'Bucuresti': (44.44, 26.1),
            'Constanta': (44.17, 28.62),
            'Iasi': (47.16, 27.58),
            'Cluj Napoca': (46.77, 23.59),
            'Timisoara': (45.75, 21.23)
        }
    
    def fit(self, X, y=None):
        if self.encoding == 'le':
            self.city_encoder = LabelEncoder() 
            self.city_encoder.fit(X['City'])
        elif self.encoding == 'ohe':
            self.city_encoder = OneHotEncoder(sparse_output=False, dtype=np.int64)
            self.city_encoder.fit(X[['City']])
        return self

    def transform(self, X):     
        if self.population:
            X['Population'] = X['City'].apply(lambda x: self.populations[x] if x in self.populations else None)
        if self.coordinates:
            X['Latitude'], X['Longitude'] = zip(*X['City'].apply(lambda x: self.city_coordinates[x]))

        if self.encoding == 'ohe':
            categ = self.city_encoder.get_feature_names_out()
            X[categ] = self.city_encoder.transform(X[['City']])
            X.drop(columns=['City'], inplace=True)
        elif self.encoding == 'le':
            X['City'] = self.city_encoder.transform(X['City'])
        else:
            X.drop(columns=['City'], inplace=True)
        return X
    
    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self
    
class Scaling(TransformerMixin):
    def __init__(self, scale_columns=[]):
        self.scale_columns = scale_columns
    
    def fit(self, X, y=None):
        if self.scale_columns:
            self.scaler = StandardScaler()
            self.scaler.fit(X[self.scale_columns])
        return self

    def transform(self, X):
        if self.scale_columns:
            for column in self.scale_columns:
                if column in X.columns:
                    X[column] = self.scaler.transform(X[[column]])
        return X
    
    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self

class MultiOutputClassifier(BaseEstimator, ClassifierMixin):
    """
    Multi target classifier that uses one classifier per target and undersamples the majority class for each target
    """
    def __init__(self, classifier=XGBClassifier, undersampler=RandomUnderSampler):
        self.classifier = classifier
        self.undersampler = undersampler

    def fit(self, X, y):
        X, y = check_X_y(X, y, multi_output=True)
        self.estimators_ = [
            self.classifier() for i in range(y.shape[1])
        ]

        for i, estimator in enumerate(self.estimators_):
            if self.undersampler:
                sampler = self.undersampler(random_state=42)
                mask = (y[:, i] == 1) | (y.sum(axis=1) == 0)
                X_res, y_res = sampler.fit_resample(X[mask], y[mask][:, i])
                estimator.fit(X_res, y_res)
            else:
                estimator.fit(X, y[:, i])
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)

        predictions = np.column_stack([estimator.predict(X) for estimator in self.estimators_])
        return predictions
    
    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)

        predictions = np.array([estimator.predict_proba(X) for estimator in self.estimators_])
        return predictions


# Final Models
    
class HospitalModel:
    def __init__(self, regressor=XGBRegressor):
        self.pipeline = Pipeline([
            ('DateTransformer', DateTransformer()),
            ('CityTransformer', CityTransformer()),
            ('Model', regressor()) 
        ])

    def fit(self, X_train, y_train):
        self.pipeline.fit(X_train, y_train)
        return self
    
    def predict(self, X):
        return self.pipeline.predict(X)
    
    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)
    
    def evaluate(self, X_test, y_test, verbose=True):

        y_pred = self.pipeline.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)

        if verbose:
            print(f"Mean Absolute Error: {mae:.3f}")
            print(f"Mean Squared Error: {mse:.3f}")

        return mae, mse
        

class PatientModel:
    def __init__(self, classifier=XGBClassifier):
        self.pipeline = Pipeline([
            ('basic_transformations', BasicTransformations()),
            ('date_transformer', DateTransformer()),
            ('classifier', MultiOutputClassifier(classifier=classifier))
        ])

    def fit(self, X_train, y_train):
        self.pipeline.fit(X_train, y_train)
        return self
    
    def predict(self, X):
        return self.pipeline.predict(X)
    
    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)
    
    def evaluate(self, X_test, y_test, verbose=True):
        target_columns = list(y_test.columns) if isinstance(y_test, pd.DataFrame) else list(y_test.name)
        results = {
            'precision': [],
            'recall': [],
            'f1': [],
            'roc_auc': []
        }
        for i, class_ in enumerate(target_columns):
            # keep only the samples that are in the current class or in the None class
            mask = (y_test.iloc[:, i] == 1) | (y_test.sum(axis=1) == 0)
            x_temp, y_temp = X_test[mask], y_test[mask]

            # undersample the majority class in order to evaluate the model
            resampler = RandomUnderSampler(random_state=42)
            X_res, y_res = resampler.fit_resample(x_temp, y_temp.iloc[:, i])
            y_pred = self.pipeline.predict(X_res)[:, i]
            y_pred_proba = self.pipeline.predict_proba(X_res)[i]

            precision = precision_score(y_res, y_pred)
            recall = recall_score(y_res, y_pred)
            f1 = f1_score(y_res, y_pred)
            roc_auc = roc_auc_score(y_res, y_pred)

            if verbose:
                print(f"\nClass: {class_}")
                print(f"Precision: {precision:.3f}")
                print(f"Recall: {recall:.3f}")
                print(f"F1 Score: {f1:.3f}")
                print(f"ROC-AUC: {roc_auc:.3f}")
            
            results['precision'].append(precision)
            results['recall'].append(recall)
            results['f1'].append(f1)
            results['roc_auc'].append(roc_auc)

        return list({key: np.mean(value) for key, value in results.items()}.values())
