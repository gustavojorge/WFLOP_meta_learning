from sklearn.model_selection import GridSearchCV

def derive_grid_from_random(best_params: dict) -> dict:
    def around_int(val, span=2, lo=1):
        vals = sorted(set([max(lo, val + d) for d in range(-span, span + 1)]))
        return vals

    grid = {}

    if 'estimator__n_estimators' in best_params:
        ne = best_params['estimator__n_estimators']
        grid['estimator__n_estimators'] = around_int(ne, span=3, lo=50)

    if 'estimator__max_depth' in best_params:
        md = best_params['estimator__max_depth']
        if md is None:
            grid['estimator__max_depth'] = [None, 2, 4, 6, 10, 15]
        else:
            grid['estimator__max_depth'] = around_int(md, span=3, lo=2) + [None]

    if 'estimator__min_samples_split' in best_params:
        mss = best_params['estimator__min_samples_split']
        grid['estimator__min_samples_split'] = around_int(mss, span=2, lo=2)

    if 'estimator__min_samples_leaf' in best_params:
        msl = best_params['estimator__min_samples_leaf']
        grid['estimator__min_samples_leaf'] = around_int(msl, span=2, lo=1)

    return grid


def run_grid_search(base_model, X, y, best_random_params, args):
    grid_params = derive_grid_from_random(best_random_params)
    grid_search = GridSearchCV(
        base_model, param_grid=grid_params, cv=args.cv,
        scoring='neg_mean_absolute_error', n_jobs=args.n_jobs, verbose=1
    )
    grid_search.fit(X, y)
    print("Best GridSearch params:", grid_search.best_params_)
    return grid_search.best_estimator_, grid_search.best_params_
