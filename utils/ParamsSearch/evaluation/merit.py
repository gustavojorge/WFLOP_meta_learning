import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import importlib.util

# Dynamic load of build_indices_dict.py
def _load_build_indices():
    components_path = Path(__file__).parents[2] / 'components' / 'build_indices_dict.py'
    spec = importlib.util.spec_from_file_location('build_indices_dict', components_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_indices_for_indicator

build_indices_for_indicator = _load_build_indices()

def compute_merit_for_model(model, df, label, n_folds=30, random_state=0):
    rng = np.random.RandomState(random_state)
    algo_cols = df.columns[-3:]
    feature_cols = df.columns[1:-3]

    indices = build_indices_for_indicator(label)
    nsga, moead, comolsd = [df[df.iloc[:, 0].astype(str).isin(indices[k])] for k in ('nsga', 'moead', 'comolsd')]
    scaler = StandardScaler()
    scores = []

    for _ in range(n_folds):
        nsga_train, nsga_test = train_test_split(nsga, test_size=0.1, random_state=rng.randint(0, 1000))
        moead_train, moead_test = train_test_split(moead, test_size=0.1, random_state=rng.randint(0, 1000))
        comolsd_train, comolsd_test = train_test_split(comolsd, test_size=0.1, random_state=rng.randint(0, 1000))

        train_data = pd.concat([nsga_train, moead_train, comolsd_train])
        test_data = pd.concat([nsga_test, moead_test, comolsd_test])

        X_train, y_train = train_data[feature_cols], train_data[algo_cols]
        X_test, y_test = test_data[feature_cols], test_data[algo_cols]
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model.fit(X_train_scaled, y_train)
        y_pred = pd.DataFrame(model.predict(X_test_scaled), columns=y_train.columns, index=y_test.index)

        AS = y_train.columns[np.argmin(y_pred.values, axis=1)]
        VBS = y_test.idxmin(axis=1)
        SBS = y_train.mean().idxmin()

        rhv_AS = np.array([y_test.loc[y_test.index[i], AS[i]] for i in range(len(X_test))])
        rhv_SBS = np.array([y_test.loc[y_test.index[i], SBS] for i in range(len(X_test))])
        rhv_VBS = np.array([y_test.at[y_test.index[i], VBS.iloc[i]] for i in range(len(X_test))])

        m = (rhv_AS.mean() - rhv_VBS.mean()) / (rhv_SBS.mean() - rhv_VBS.mean())
        scores.append(m)

    return float(np.mean(scores)) if scores else float('inf')
