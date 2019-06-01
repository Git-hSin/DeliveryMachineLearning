import numpy as np
import model_main as m

train_features = m.X_train
train_labels =  m.y_train
test_features = m.X_test
test_labels = m.y_test

def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return accuracy

from sklearn.ensemble import RandomForestRegressor

base_model = RandomForestRegressor(n_estimators = 10, random_state = 42)
base_model.fit(train_features, train_labels)
base_accuracy = evaluate(base_model, test_features, test_labels)

from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search 
#           max_features='auto', max_leaf_nodes=None,
#           min_impurity_decrease=0.0, min_impurity_split=None,
#           min_samples_leaf=4, min_samples_split=2,
#           min_weight_fraction_leaf=0.0, n_estimators=600, n_jobs=1,
#           oob_score=False, random_state=None, verbose=0, warm_start=False)


param_grid = {
    'bootstrap': [False],
    'max_depth': [80, 90, 100, 110],
    'max_features': [60, 80, 100],
    'min_samples_leaf': [1, 2, 3],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [10, 200, 300, 600]
}
# Create a based model

rf = RandomForestRegressor()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)


grid_search.fit(train_features, train_labels)
grid_search.best_params_
best_grid = grid_search.best_estimator_
grid_accuracy = evaluate(best_grid, test_features, test_labels)
print('Improvement of {:0.2f}%.'.format( 100 * (grid_accuracy - base_accuracy) / base_accuracy))
#rf = RandomForestRegressor(n_estimators=200)
#rf = rf.fit(X_train, y_train)
#rf.score(X_test, y_test)

#sorted(zip(rf.feature_importances_, feature_names), reverse=True)
