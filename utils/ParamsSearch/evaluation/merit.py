import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import importlib.util

try:
    from imblearn.over_sampling import SMOTE, RandomOverSampler  # type: ignore
except ImportError:
    SMOTE = None  # type: ignore
    RandomOverSampler = None  # type: ignore


# === Dynamic load of indices ===
def _load_build_indices():
    components_path = Path(__file__).parents[3] / 'components' / 'build_indices_dict.py'
    spec = importlib.util.spec_from_file_location('build_indices_dict', components_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_indices_for_indicator

build_indices_for_indicator = _load_build_indices()


# === Merit computation consistent with models_and_merit_builder ===
def compute_merit_for_model(model, df, label, n_folds=30, random_state=0):
    print("Folds for merit estimation:", n_folds)

    algo_cols = df.columns[-3:]
    feature_cols = df.columns[1:-3]
    indices = build_indices_for_indicator(label)
    nsga, moead, comolsd = [df[df.iloc[:, 0].astype(str).isin(indices[k])] for k in ('nsga', 'moead', 'comolsd')]
    scaler = StandardScaler()
    rng = np.random.RandomState(random_state)
    scores: list[float] = []

    # === Define test sizes per label (same as builder) ===
    if label == "epsilon":
        size_nsga_test = 3
        size_moead_test = 1
    elif label == "hipervolume":
        size_nsga_test = 1
        size_moead_test = 4
    elif label == "igd":
        size_nsga_test = 4
        size_moead_test = 7
    else:
        raise ValueError("Label desconhecido para tamanhos de teste")

    def _instance_minmax(df_algos: pd.DataFrame) -> pd.DataFrame:
        row_min = df_algos.min(axis=1)
        row_max = df_algos.max(axis=1)
        span = (row_max - row_min).replace(0, 1)
        return (df_algos.sub(row_min, axis=0)).div(span, axis=0)

    for fold in range(n_folds):
        # === Seleção manual como no models_and_merit_builder ===
        if len(nsga) < size_nsga_test or len(moead) < size_moead_test:
            raise ValueError("Instâncias insuficientes para o tamanho de teste definido.")

        nsga_test_idx = np.random.choice(nsga.iloc[:, 0], size=size_nsga_test, replace=False)
        moead_test_idx = np.random.choice(moead.iloc[:, 0], size=size_moead_test, replace=False)

        nsga_train = nsga[~nsga.iloc[:, 0].isin(nsga_test_idx)]
        nsga_test = nsga[nsga.iloc[:, 0].isin(nsga_test_idx)]
        moead_train = moead[~moead.iloc[:, 0].isin(moead_test_idx)]
        moead_test = moead[moead.iloc[:, 0].isin(moead_test_idx)]

        comolsd_train, comolsd_test = train_test_split(
            comolsd, test_size=0.1, random_state=rng.randint(0, 1000)
        )

        train_data = pd.concat([nsga_train, moead_train, comolsd_train])
        test_data = pd.concat([nsga_test, moead_test, comolsd_test])

        X_train, y_train_raw = train_data[feature_cols], train_data[algo_cols]
        X_test, y_test_raw = test_data[feature_cols], test_data[algo_cols]

        # === Class vector for SMOTE ===
        train_ids = train_data.iloc[:, 0].astype(str)
        y_classes = []
        for inst_id in train_ids:
            if inst_id in indices['nsga']:
                y_classes.append(0)
            elif inst_id in indices['moead']:
                y_classes.append(1)
            elif inst_id in indices['comolsd']:
                y_classes.append(2)
        y_classes = np.array(y_classes)

        # === Moderate oversampling ===
        if SMOTE is not None and RandomOverSampler is not None:
            unique, counts = np.unique(y_classes, return_counts=True)
            max_count = counts.max()
            target_minor = int(max_count * 0.85)
            sampling_strategy = {u: (target_minor if c < target_minor else c) for u, c in zip(unique, counts)}
            min_count = counts.min()
            combined = pd.concat([X_train.reset_index(drop=True), y_train_raw.reset_index(drop=True)], axis=1)

            if min_count < 3:
                ros = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=0)
                combined_res, y_classes_res = ros.fit_resample(combined, y_classes)
            else:
                k_neighbors = max(1, min(min_count - 1, 5))
                smote = SMOTE(sampling_strategy=sampling_strategy, k_neighbors=k_neighbors, random_state=0)
                combined_res, y_classes_res = smote.fit_resample(combined, y_classes)

            X_train = combined_res.iloc[:, :len(feature_cols)]
            y_train_raw = combined_res.iloc[:, len(feature_cols):]
            y_train_raw.columns = algo_cols

        # === Normalizações ===
        y_train = _instance_minmax(y_train_raw)
        y_test = _instance_minmax(y_test_raw)

        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # === Avaliação ===
        model.fit(X_train_scaled, y_train)
        y_pred = pd.DataFrame(model.predict(X_test_scaled), columns=y_train.columns, index=y_test.index)

        AS = y_train.columns[np.argmin(y_pred.values, axis=1)]
        VBS = y_test.idxmin(axis=1)
        SBS = y_train.mean().idxmin()

        rhv_AS = np.array([y_test.loc[y_test.index[i], AS[i]] for i in range(len(X_test))])
        rhv_SBS = np.array([y_test.loc[y_test.index[i], SBS] for i in range(len(X_test))])
        rhv_VBS = np.array([y_test.at[y_test.index[i], VBS.iloc[i]] for i in range(len(X_test))])

        denom = rhv_SBS.mean() - rhv_VBS.mean()
        if denom != 0:
            m = (rhv_AS.mean() - rhv_VBS.mean()) / denom
            scores.append(m)

    return float(np.mean(scores)) if scores else float('inf')
