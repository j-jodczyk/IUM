import inspect
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
import pickle
import logging
from load_data import Preprocessor, DataModel


class NaiveModel:
    def fit(self, X, y):
        return None
    
    def predict(self, input_df):
        user_ids = input_df.index
        mock_series = pd.Series(True, index=user_ids, name="user_id")
        return mock_series


class ModelManager:
    def __init__(self, model=KNeighborsClassifier):
        self.model = model() if inspect.isclass(model) else model
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_hat = None

    # Shouldn't be in the model
    def prepare_data(self, since=None) :
        logging.info(f"Enter: prepare_data")
        data_paths = {"users_path": "../data_jsonl/users.jsonl", "tracks_path":"../data_jsonl/tracks.jsonl", "artists_path":"../data_jsonl/artists.jsonl", "sessions_path":"../data_jsonl/sessions.jsonl"}
        data_model = DataModel(load_data=True, data_paths_dict=data_paths)
        data_df = data_model.get_merged_dfs(since=since)
        # TODO:check why since works?
        self.X_train, self.X_test, self.y_train, self.y_test = Preprocessor(data_df).transform(split=True)
        
        return self

    def fit_data(self, X_train, Y_train):
        self.y_hat = None
        self.model.fit(X_train, Y_train)
        return self

    def fit(self, X_train, Y_train):
        return self.fit_data(X_train, Y_train)

    def fit_data(self):
        self.y_hat = None
        self.model.fit(self.X_train, self.y_train)
        return self

    def predict(self, X_test):
        self.y_hat = self.model.predict(X_test)
        return self.y_hat

    def predict(self):
        self.y_hat = self.model.predict(self.X_test)
        return self.y_hat

    def accuracy(self):
        if self.y_hat is None:
            return None
        return accuracy_score(self.y_test, self.y_hat)

    def classification_report(self):
        if self.y_hat is None:
            return None
        return classification_report(self.y_test, self.y_hat)

    def save_model_to_file(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self.model, f)

    def get_params(self):
        return self.model.get_params()
