from models import ModelManager
import numpy as np

# TODO: add this everywhere instead of passing random_state param - or make some main apps
np.random.seed(31415)
def grid_search_params():
    model_manager = ModelManager()
    model_manager.prepare_data(since=np.datetime64('2019-08', 'D')).fit_data().predict()
    print(model_manager.classification_report())
    
if __name__ == "__main__":
    grid_search_params()