from load_data import Preprocessor
import pickle


class NaiveModel:
    def predict(self, input_data):
        return [1]


class ModelManager:
    def __init__(self, model):
        self.model = model()

    def prepare_data(self):
        data_paths = ["users.json", "tracks.json", "artists.json", "sessions.json"]
        # TODO change
        preprocessor = Preprocessor(data_paths)
        data_df = preprocessor.run()

        self.X = data_df.drop(["premium_user", "street", "name", "user_id"], axis=1)
        self.y = data_df.loc[:, "premium_user"]

    def fit_data(self, X_train, Y_train):
        self.model.fit(self, X_train, Y_train)

    def predict(self, X_test):
        y_hat = self.model.predict(X_test)
        return y_hat

    def save_model_to_file(self, filename):
        pickle.dump(self.model, open(filename, "wb"))
