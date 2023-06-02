from models import ModelManager
import numpy as np


def train_model():
    model_manager = ModelManager()
    model_manager.prepare_data(since=np.datetime64('2019-08', 'D')).fit_data().predict()
    print(model_manager.classification_report())
    
if __name__ == "__main__":
    train_model()