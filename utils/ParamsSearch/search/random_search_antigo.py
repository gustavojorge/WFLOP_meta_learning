from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

def run_random_search(base_model, X, y, args):
    param_dist = {
        'estimator__n_estimators': randint(100, 1000),
        'estimator__max_depth': [None, 2, 4, 10, 20, 25],
        'estimator__min_samples_split': randint(2, 10),
        'estimator__min_samples_leaf': randint(1, 5)
    }

    rand_search = RandomizedSearchCV(
        base_model, param_distributions=param_dist, n_iter=args.random_iters,
        cv=args.cv, scoring='neg_mean_absolute_error', n_jobs=args.n_jobs,
        random_state=0, verbose=1
    )
    rand_search.fit(X, y)
    print("Best RandomSearch params:", rand_search.best_params_)
    return rand_search.best_estimator_, rand_search.best_params_