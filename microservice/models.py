from load_data import Preprocessor, DataModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle


class NaiveModel:
    def predict(self, input_data):
        return [1]


class ModelManager:
    def __init__(self, model=NaiveModel):
        self.model = model()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_hat = None
        
    # Shouldn't be in the model
    def prepare_data(self) :
        data_paths = {"users_path": "./data_jsonl/users.jsonl", "tracks_path":"./data_jsonl/tracks.jsonl", "artists_path":"./data_jsonl/artists.jsonl", "sessions_path":"./data_jsonl/sessions.jsonl"}
        data_model = DataModel(load_data=True, data_paths_dict=data_paths)
        data_df = data_model.get_merged_dfs()
        data_df = Preprocessor.run(data_df)
        X = data_df.drop(["premium_user"], axis=1)
        y = data_df.loc[:, "premium_user"]
        def split_data(test_size:float = 0.3):
            # TODO: here radom_state or global random_state
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size)
        split_data()
        return self

    def fit_data(self, X_train, Y_train):
        self.y_hat = None
        self.model.fit(self, X_train, Y_train)
        return self

    def fit_data(self):
        self.y_hat = None
        self.model.fit(self,self.X_train, self.y_train)
        return self
    
    def predict(self, X_test):
        self.y_hat = self.model.predict(X_test)
        return self.y_hat

    def predict(self):
        y_hat = self.model.predict(self.X_test)
        return y_hat

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
