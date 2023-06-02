from load_data import Preprocessor, DataModel
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
import pickle
import logging
import os

# defining log file:
def config_logging(number: int) -> None:
    # create a directory for logging
    log_dir = "log"
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    logging.basicConfig(
        level=logging.DEBUG,
        format="{asctime} {levelname:<8} {message}",
        style="{",
        filename="./log/%d.log" % number,
        filemode="a",
        force=True,
    )

config_logging(0)

class NaiveModel:
    def fit(self, X, y):
        return None
    
    def predict(self, input_df):
        user_ids = input_df.index
        mock_series = pd.Series(True, index=user_ids, name="user_id")
        return mock_series


class ModelManager:
    def __init__(self, model=KNeighborsClassifier):
        self.model = model()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_hat = None

    # Shouldn't be in the model
    def prepare_data(self) :
        # TODO remove - its for testing
        # LinesOfData=50000
        
        logging.info(f"Enter: prepare_data")
        data_paths = {"users_path": "../data_jsonl/users.jsonl", "tracks_path":"../data_jsonl/tracks.jsonl", "artists_path":"../data_jsonl/artists.jsonl", "sessions_path":"../data_jsonl/sessions.jsonl"}
        data_model = DataModel(load_data=True, data_paths_dict=data_paths)
        # TODO remove LinesOfData - its for testing
        # data_df = data_model.get_merged_dfs(N=LinesOfData)
        data_df = data_model.get_merged_dfs()
        data_df = Preprocessor.run(data_df)
        X = data_df.drop(["premium_user"], axis=1)
        y = data_df.loc[:, "premium_user"]

        def split_data(test_size: float = 0.3):
            # TODO: here radom_state or global random_state
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size
            )

        split_data()
        return self

    def fit_data(self, X_train, Y_train):
        logging.info(f"Enter: fit_data")
        self.y_hat = None
        self.model.fit( X_train, Y_train)
        return self

    def fit_data(self):
        logging.info(f"Enter: fit_data with \tX_train len: {len(self.X_train.index)} \ty_train len: {len(self.y_train.index)}")
        self.y_hat = None
        self.model.fit(self.X_train, self.y_train)
        return self

    def predict(self, X_test):
        logging.info(f"Enter: predict with \tX_test len: {len(X_test.index)}")
        self.y_hat = self.model.predict(X_test)
        logging.info(f"Left: predict with \ty_hat len: {len(self.y_hat.index)}")
        return self.y_hat

    def predict(self):
        logging.info(f"Enter: predict with \tX_test len: {len(self.X_test.index)}")
        self.y_hat = self.model.predict(self.X_test)
        logging.info(f"Left: predict with \ty_hat len: {len(self.y_hat.index)}")
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
        pickle.dump(self.model, open(filename, "wb"))
