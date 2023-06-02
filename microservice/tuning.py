from sklearn.model_selection import ShuffleSplit
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from models import ModelManager, config_logging
from load_data import Preprocessor, DataModel
import numpy as np
import logging

config_logging("tuning")

class ParametersSearchLoop (object):
    def __init__ (self, default_model=True, prepare_data=True, estimator_params:dict=None, scoring='f1_micro', n_jobs=None):
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.estimator_params:dict = dict()
        self.X = None
        self.y = None
        
        if default_model:
            self.set_default_models()
        else:
            self.estimator_params:dict = estimator_params
        if prepare_data:
            self.prepare_data()    
        

    def set_default_models(self):
        def get_kneigh():
            leaf_size = list(range(1,20))
            # leaf_size = list(range(1,50))
            n_neighbors = list(range(1,10))
            # n_neighbors = list(range(1,30))
            p=[1,2]
            # weights=['uniform', 'distance']
            # algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']
            kneigh_params = {
                'leaf_size': leaf_size, 
                'n_neighbors': n_neighbors, 
                'p':p
                }
            # kneigh_params = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p, weights=weights, algorithm=algorithm)
            return { ModelManager(KNeighborsClassifier): kneigh_params}
        
        def get_tree():
            max_depth = [2, 3, 5, 10, 20]
            min_samples_leaf = [5, 10, 20, 50, 100]
            criterion = ["gini", "entropy"]
            
            tree_params = {
                'max_depth': max_depth,
                'min_samples_leaf': min_samples_leaf,
                'criterion': criterion
            }
            return { ModelManager(DecisionTreeClassifier): tree_params}
        
        def get_forest():
            n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 5)]
            max_features = ['auto', 'sqrt']
            max_depth = [int(x) for x in np.linspace(10, 110, num = 5)]
            max_depth.append(None)
            min_samples_split = [2, 5, 10]
            min_samples_leaf = [1, 2, 4]
            bootstrap = [True, False]
            
            forest_params = {'n_estimators': n_estimators,
                        'max_features': max_features,
                        'max_depth': max_depth,
                        'min_samples_split': min_samples_split,
                        'min_samples_leaf': min_samples_leaf,
                        'bootstrap': bootstrap}
            
            return { ModelManager(RandomForestClassifier): forest_params}
        self.estimator_params.update(get_kneigh())
        self.estimator_params.update(get_tree())
        self.estimator_params.update(get_forest())

# TOOD: unify between this and modelmanager
    def prepare_data(self, 
            data_paths = {"users_path": "./data_jsonl/users.jsonl", "tracks_path":"./data_jsonl/tracks.jsonl", "artists_path":"./data_jsonl/artists.jsonl", "sessions_path":"./data_jsonl/sessions.jsonl"}
                    ):
        data_model = DataModel(load_data=True, data_paths_dict=data_paths)
        data_df = data_model.get_merged_dfs()
        data_df = Preprocessor.transform(data_df, split=False)
        self.X = data_df.drop(["premium_user"], axis=1)
        self.y = data_df.loc[:, "premium_user"]

    # TODO: change to __next__
    # TODO: add verbose and PArallel
    def grid_generator(self):
        for model_manager, params in self.estimator_params.items():
            # TODO: choose scoring, change to model_manager back
            # model_manager can be passed directly to the grid, so it will log values and update accuracy if set up correctly, but due to time consumption it is not configured that way 
            grid = HalvingRandomSearchCV(model_manager.model, params, scoring=self.scoring, cv=ShuffleSplit(
                                    test_size=0.3, n_splits=5, random_state=42),
                                verbose=2, n_jobs=-1, n_candidates="exhaust", factor=3
                                    )
            try:   
                grid.fit(self.X, self.y)
                logging.info(f"Best params: {grid.best_params_}\tBest score: {grid.best_score_}\tBest Estimator: {grid.best_estimator_}")
            except KeyboardInterrupt:
                message = f"PROGRAM STOPPED: Best params: {grid.best_params_}\tBest score: {grid.best_score_}\tBest Estimator: {grid.best_estimator_}"
                logging.info(message)
                print(message)

            yield grid.best_params_, grid.best_score_, grid.best_estimator_
                
if __name__ == '__main__':
    grid = ParametersSearchLoop()
    for grid_params in grid.grid_generator():
        print(f"Params: {grid_params[0]}, \t score: {grid_params[1]}")