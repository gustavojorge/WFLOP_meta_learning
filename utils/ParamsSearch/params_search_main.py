import argparse
import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import randint

# Reuse dynamic indices logic
def _load_build_indices():
    import sys
    import importlib.util
    components_path = Path(__file__).parents[2] / 'components' / 'build_indices_dict.py'
    spec = importlib.util.spec_from_file_location('build_indices_dict', components_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_indices_for_indicator

build_indices_for_indicator = _load_build_indices()

def compute_merit_for_model(model: MultiOutputRegressor, df: pd.DataFrame, label: str, n_folds: int = 30,
                             size_nsga_test: int = None, size_moead_test: int = None, random_state: int = 0) -> float:

    rng = np.random.RandomState(random_state)
    # Determine algorithm columns (last 3) and features (excluding first id column & last 3)
    algo_cols = df.columns[-3:]
    feature_cols = df.columns[1:-3]

    indices = build_indices_for_indicator(label)
    nsga_indices = indices['nsga']
    moead_indices = indices['moead']
    comolsd_indices = indices['comolsd']

    nsga_instances = df[df.iloc[:, 0].astype(str).isin(nsga_indices)]
    moead_instances = df[df.iloc[:, 0].astype(str).isin(moead_indices)]
    comolsd_instances = df[df.iloc[:, 0].astype(str).isin(comolsd_indices)]

    # Default test sizes (copied from existing logic) if not given
    if size_nsga_test is None or size_moead_test is None:
        if label == 'epsilon':
            size_nsga_test, size_moead_test = 3, 1
        elif label == 'hipervolume':
            size_nsga_test, size_moead_test = 1, 4
        elif label == 'igd':
            size_nsga_test, size_moead_test = 4, 7
        else:
            raise ValueError('Unknown label for test sizes')

    scaler = StandardScaler()
    scores = []

    for _ in range(n_folds):
        if len(nsga_indices) < size_nsga_test or len(moead_indices) < size_moead_test:
            raise ValueError('Not enough indices for requested test sizes')

        selected_nsga_test = rng.choice(nsga_indices, size=size_nsga_test, replace=False)
        selected_moead_test = rng.choice(moead_indices, size=size_moead_test, replace=False)

        nsga_train = nsga_instances[~nsga_instances.iloc[:, 0].astype(str).isin(selected_nsga_test)]
        nsga_test = nsga_instances[nsga_instances.iloc[:, 0].astype(str).isin(selected_nsga_test)]
        moead_train = moead_instances[~moead_instances.iloc[:, 0].astype(str).isin(selected_moead_test)]
        moead_test = moead_instances[moead_instances.iloc[:, 0].astype(str).isin(selected_moead_test)]

        comolsd_train, comolsd_test = train_test_split(comolsd_instances, test_size=0.1, random_state=rng.randint(0, 100000))

        train_data = pd.concat([nsga_train, moead_train, comolsd_train])
        test_data = pd.concat([nsga_test, moead_test, comolsd_test])

        X_train, y_train = train_data.loc[:, feature_cols], train_data.loc[:, algo_cols]
        X_test, y_test = test_data.loc[:, feature_cols], test_data.loc[:, algo_cols]

        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Fit provided model clone each fold (fresh clone to avoid leakage)
        model.fit(X_train_scaled, y_train)
        y_pred = pd.DataFrame(model.predict(X_test_scaled), columns=y_train.columns, index=y_test.index)

        AS = y_train.columns[np.argmin(y_pred.values, axis=1)]
        VBS = y_test.idxmin(axis=1)
        SBS = y_train.mean().idxmin()

        rhv_AS = np.array([y_test.loc[y_test.index[i], AS[i]] for i in range(len(X_test))])
        rhv_SBS = np.array([y_test.loc[y_test.index[i], SBS] for i in range(len(X_test))])
        rhv_VBS = np.array([y_test.at[y_test.index[i], VBS.iloc[i]] for i in range(len(X_test))])

        rhv_AS_mean = rhv_AS.mean()
        rhv_SBS_mean = rhv_SBS.mean()
        rhv_VBS_mean = rhv_VBS.mean()
        if rhv_SBS_mean == rhv_VBS_mean:
            continue  # skip degenerate fold
        m = (rhv_AS_mean - rhv_VBS_mean) / (rhv_SBS_mean - rhv_VBS_mean)
        scores.append(m)

    return float(np.mean(scores)) if scores else float('inf')

def derive_grid_from_random(best_params: dict) -> dict:
    """Expand best params into a focused grid for fine tuning."""
    grid = {}
    # Parameters come with prefix 'estimator__'
    def around_int(val, span=2, lo=1):
        vals = sorted(set([max(lo, val + d) for d in range(-span, span + 1)]))
        return vals

    # n_estimators
    if 'estimator__n_estimators' in best_params:
        ne = best_params['estimator__n_estimators']
        grid['estimator__n_estimators'] = around_int(ne, span=3, lo=50)
    # max_depth can be None or int
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

def main():
    parser = argparse.ArgumentParser(description='Param search orchestrator (Random -> Grid) minimizing merit m.')
    parser.add_argument('--label', required=True, choices=['epsilon','hipervolume','igd'], help='Indicator label')
    parser.add_argument('--l', required=True, type=int, help='Length l value (e.g., 10)')
    parser.add_argument('--r', required=True, type=str, help='Neighborhood r value string (e.g., 1.0)')
    parser.add_argument('--random-iters', type=int, default=20, help='RandomizedSearchCV iterations')
    parser.add_argument('--folds-merit', type=int, default=30, help='Folds to approximate merit m')
    parser.add_argument('--cv', type=int, default=3, help='CV folds for search procedures')
    parser.add_argument('--n-jobs', type=int, default=-1, help='Parallel jobs for search')
    parser.add_argument('--output', type=str, default=None, help='Optional JSON output path for results')
    args = parser.parse_args()

    dataset_path = f'meta_dataset/{args.label}/pareto_based/random_walk/l{args.l}_r{args.r}.csv'
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f'Dataset not found: {dataset_path}')
    df = pd.read_csv(dataset_path)

    algo_cols = df.columns[-3:]
    feature_cols = df.columns[1:-3]

    X = df.loc[:, feature_cols]
    y = df.loc[:, algo_cols]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    base = MultiOutputRegressor(RandomForestRegressor(random_state=0))

    param_dist = {
        'estimator__n_estimators': randint(100, 1200),
        'estimator__max_depth': [None, 2, 4, 10, 20, 25],
        'estimator__min_samples_split': randint(2, 10),
        'estimator__min_samples_leaf': randint(1, 5)
    }

    print('=== Random Search Phase ===')
    rand_search = RandomizedSearchCV(base, param_distributions=param_dist, n_iter=args.random_iters,
                                     cv=args.cv, scoring='neg_mean_absolute_error', n_jobs=args.n_jobs,
                                     random_state=0, verbose=1)
    rand_search.fit(X_scaled, y)
    best_random_params = rand_search.best_params_
    print('Best random params:', best_random_params)

    # Merit with random best
    random_best_model = rand_search.best_estimator_
    merit_random = compute_merit_for_model(random_best_model, df, args.label, n_folds=args.folds_merit)
    print(f'Approx merit (random best): {merit_random:.5f}')

    grid_params = derive_grid_from_random(best_random_params)
    print('Derived grid for refinement:', grid_params)

    print('=== Grid Search Phase ===')
    grid_search = GridSearchCV(base, param_grid=grid_params, cv=args.cv, scoring='neg_mean_absolute_error',
                               n_jobs=args.n_jobs, verbose=1)
    grid_search.fit(X_scaled, y)
    best_grid_params = grid_search.best_params_
    print('Best grid params:', best_grid_params)
    grid_best_model = grid_search.best_estimator_
    merit_grid = compute_merit_for_model(grid_best_model, df, args.label, n_folds=args.folds_merit)
    print(f'Approx merit (grid best): {merit_grid:.5f}')

    results = {
        'dataset': dataset_path,
        'label': args.label,
        'l': args.l,
        'r': args.r,
        'random_search': {
            'best_params': best_random_params,
            'approx_merit': merit_random
        },
        'grid_search': {
            'best_params': best_grid_params,
            'approx_merit': merit_grid
        }
    }

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f'Results saved to {out_path}')
    else:
        print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()
