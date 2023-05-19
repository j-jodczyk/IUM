from microservice.load_data import Preprocessor


class Model:
    def __init__(self, model):
        self.model = model()

    def prepare_data(self):
        data_paths = ["users.json", "tracks.json", "artists.json", "sessions.json"]
        preprocessor = Preprocessor(data_paths)
        data_df = preprocessor.run()

        self.X = data_df.drop(["premium_user", "street", "name", "user_id"], axis=1)
        self.y = data_df.loc[:, "premium_user"]

    def fit_data(self, X_train, Y_train):
        self.model.fit(self, X_train, Y_train)

    def predict(self, X_test):
        y_hat = self.model.predict(X_test)
        return y_hat

    def save_to_file(self):
        # TODO
        pass


class NaiveModel:
    def predict(self, input_data):
        return [1]
