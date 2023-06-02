from sklearn.model_selection import GridSearchCV, ShuffleSplit

vectorizer_param_grid = { 'max_df': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0], 'vect__min_df': [1, 2, 3, 0.05, 0.1, 0.15, 0.2], 
                            'metric': ['cosine', 'euclidean']}


grid = GridSearchCV(pipeline, vectorizer_param_grid, scoring='f1_micro', cv=ShuffleSplit(
                        test_size=0.3, n_splits=5),
                        )
grid.fit(train_data,y)
grid.best_params_

