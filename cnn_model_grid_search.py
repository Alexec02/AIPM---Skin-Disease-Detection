from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV
from cnn_model import create_model

def perform_grid_search(X_train, y_train):
    model = KerasClassifier(build_fn=create_model, verbose=0)  
    
    param_grid = {
        'optimizer': ['adam', 'sgd'], 
        # 'dropout_rate': [0.3, 0.5, 0.7],
        'batch_size': [32, 64],
        'epochs': [10, 20]
    }

    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='accuracy')
    grid_result = grid.fit(X_train, y_train)

    print(f"Best parameters: {grid_result.best_params_}")
    print(f"Best score: {grid_result.best_score_}")

    return grid.best_estimator_
