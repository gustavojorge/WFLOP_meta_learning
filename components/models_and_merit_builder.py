import argparse
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
try:
    from imblearn.over_sampling import SMOTE, RandomOverSampler  # type: ignore
except ImportError:
    SMOTE = None  # type: ignore
    RandomOverSampler = None  # type: ignore

# Importa indices dinâmicos
try:
    from .build_indices_dict import build_all_indices  # type: ignore
except ImportError:
    import os as _os, sys as _sys
    _pkg_root = _os.path.dirname(_os.path.abspath(__file__))
    _project_root = _os.path.dirname(_pkg_root)
    if _project_root not in _sys.path:
        _sys.path.insert(0, _project_root)
    from build_indices_dict import build_all_indices  # type: ignore

def models_and_merit_builder(file_path, arquive, label):
    output_path = f'result/{label}/features_importance/'
    models_merit_path = f'result/{label}/models/theoretical_models/{arquive}/'
    models_pickle_path = f'result/{label}/models/pickle_models/{arquive}/'
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(models_merit_path, exist_ok=True)
    os.makedirs(models_pickle_path, exist_ok=True)

    df = pd.read_csv(file_path)

    # ------------ INDICES DINÂMICOS ------------
    all_indices = build_all_indices()
    label_indices = all_indices[label]

    nsga_indices = label_indices["nsga"]
    moead_indices = label_indices["moead"]
    comolsd_indices = label_indices["comolsd"]

    nsga_instances = df[df.iloc[:, 0].astype(str).isin(nsga_indices)]
    moead_instances = df[df.iloc[:, 0].astype(str).isin(moead_indices)]
    comolsd_instances = df[df.iloc[:, 0].astype(str).isin(comolsd_indices)]

    scaler = StandardScaler()
    n_folds = 100
    scores = []

    algo_cols = df.columns[-3:]
    feature_cols = df.columns[1:-3]
    feature_importances = np.zeros(len(feature_cols))

    best_model = None
    worst_model = None
    best_score = float('inf')
    worst_score = float('-inf')

    as_predictions = []
    vbs_predictions = []
    sbs_predictions = []
    as_predictions_raw = []
    vbs_predictions_raw = []
    sbs_predictions_raw = []

    sbs_best_model = []
    fold_numbers = []

    # Custom fold sampling sizes
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
        raise ValueError("Label desconhecido")

    def _instance_minmax(df_algos):
        row_min = df_algos.min(axis=1)
        row_max = df_algos.max(axis=1)
        span = (row_max - row_min).replace(0, 1)
        return (df_algos.sub(row_min, axis=0)).div(span, axis=0)

    print("[InstanceNorm] Min-max por instância aplicado nas colunas de algoritmos...")

    for fold_number in range(n_folds):
        print(f"Fold {fold_number + 1}/{n_folds}")
        fold_numbers.append(fold_number)

        # Split respecting dynamic indices
        selected_nsga_indices_test = np.random.choice(nsga_indices, size=size_nsga_test, replace=False)
        selected_moead_indices_test = np.random.choice(moead_indices, size=size_moead_test, replace=False)

        selected_nsga_indices_train = list(set(nsga_indices) - set(selected_nsga_indices_test))
        selected_moead_indices_train = list(set(moead_indices) - set(selected_moead_indices_test))

        nsga_train = nsga_instances[nsga_instances.iloc[:, 0].astype(str).isin(selected_nsga_indices_train)]
        nsga_test = nsga_instances[nsga_instances.iloc[:, 0].astype(str).isin(selected_nsga_indices_test)]
        moead_train = moead_instances[moead_instances.iloc[:, 0].astype(str).isin(selected_moead_indices_train)]
        moead_test = moead_instances[moead_instances.iloc[:, 0].astype(str).isin(selected_moead_indices_test)]

        comolsd_train, comolsd_test = train_test_split(comolsd_instances, test_size=0.1, random_state=np.random.randint(0, 1000))

        train_data = pd.concat([nsga_train, moead_train, comolsd_train])
        test_data = pd.concat([nsga_test, moead_test, comolsd_test])

        X_train, y_train_raw = train_data.loc[:, feature_cols], train_data.loc[:, algo_cols]
        X_test, y_test_raw = test_data.loc[:, feature_cols], test_data.loc[:, algo_cols]

        # ---------- Store ORIGINAL y_target BEFORE SMOTE for SBS ----------
        y_train_raw_original = y_train_raw.copy()

        # Build y_classes
        train_ids = train_data.iloc[:, 0].astype(str)
        y_classes = np.array([
            0 if i in nsga_indices else 1 if i in moead_indices else 2 for i in train_ids
        ])

        unique, counts = np.unique(y_classes, return_counts=True)
        max_count = counts.max()
        target_minor = int(max_count * 0.80)
        sampling_strategy = {u: (target_minor if c < target_minor else c) for u, c in zip(unique, counts)}

        if SMOTE is not None and RandomOverSampler is not None:
            if counts.min() < 3:
                ros = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=0)
                combined = pd.concat([X_train.reset_index(drop=True), y_train_raw.reset_index(drop=True)], axis=1)
                combined_res, y_classes_res = ros.fit_resample(combined, y_classes)
            else:
                k_neighbors = max(1, min(counts.min() - 1, 5))
                smote = SMOTE(sampling_strategy=sampling_strategy, k_neighbors=k_neighbors, random_state=0)
                combined = pd.concat([X_train.reset_index(drop=True), y_train_raw.reset_index(drop=True)], axis=1)
                combined_res, y_classes_res = smote.fit_resample(combined, y_classes)

            X_train = combined_res.iloc[:, :len(feature_cols)]
            y_train_raw = combined_res.iloc[:, len(feature_cols):]
            y_train_raw.columns = algo_cols

        # Normalize y AFTER SMOTE (for AS)
        y_train = _instance_minmax(y_train_raw)
        y_test = _instance_minmax(y_test_raw)

        # ✅ Normalize ORIGINAL y for SBS baseline (before SMOTE)
        y_train_original_norm = _instance_minmax(y_train_raw_original)

        # Scale X
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = MultiOutputRegressor(RandomForestRegressor(
            n_estimators=498, max_depth=4, min_samples_split=2, min_samples_leaf=4,
            random_state=0, n_jobs=-1
        ))

        #model = MultiOutputRegressor(RandomForestRegressor(random_state=0, n_jobs=-1))

        model.fit(X_train_scaled, y_train)

        feature_importances += np.mean([est.feature_importances_ for est in model.estimators_], axis=0)

        y_pred = pd.DataFrame(model.predict(X_test_scaled), columns=y_train.columns, index=y_test.index)

        AS = y_train.columns[np.argmin(y_pred.values, axis=1)]
        VBS = y_test.idxmin(axis=1)

        # SBS from ORIGINAL TRAIN (pre-SMOTE)
        SBS = y_train_original_norm.mean().idxmin()

        sbs_best_model.append(str(SBS))

        rhv_AS = np.array([y_test.loc[i, AS[j]] for j, i in enumerate(y_test.index)])
        rhv_SBS = np.array([y_test.loc[i, SBS] for i in y_test.index])
        rhv_VBS = np.array([y_test.at[i, VBS.loc[i]] for i in y_test.index])

        rhv_AS_raw = np.array([y_test_raw.loc[i, AS[j]] for j, i in enumerate(y_test.index)])
        rhv_SBS_raw = np.array([y_test_raw.loc[i, SBS] for i in y_test.index])
        rhv_VBS_raw = np.array([y_test_raw.at[i, VBS.loc[i]] for i in y_test.index])

        as_predictions.append(rhv_AS)
        vbs_predictions.append(rhv_VBS)
        sbs_predictions.append(rhv_SBS)
        as_predictions_raw.append(rhv_AS_raw)
        vbs_predictions_raw.append(rhv_VBS_raw)
        sbs_predictions_raw.append(rhv_SBS_raw)

        rhv_AS_mean = rhv_AS.mean()
        rhv_SBS_mean = rhv_SBS.mean()
        rhv_VBS_mean = rhv_VBS.mean()

        if rhv_SBS_mean == rhv_VBS_mean:
            continue

        m = (rhv_AS_mean - rhv_VBS_mean) / (rhv_SBS_mean - rhv_VBS_mean)
        scores.append(m)

        if m < best_score:
            best_score = m
            best_model = model

        if m > worst_score:
            worst_score = m
            worst_model = model

    feature_importances /= n_folds
    pd.DataFrame({'Feature': feature_cols, 'Importance': feature_importances}) \
        .sort_values(by='Importance', ascending=False) \
        .to_csv(f"{output_path}{arquive}_features_importance.csv", index=False)

    print(np.mean(scores))
    print([float(x) for x in scores])

    # ----- Final model full training -----
    X, y_raw_full = df.loc[:, feature_cols], df.loc[:, algo_cols]
    y = _instance_minmax(y_raw_full)
    scaler.fit(X)
    X_scaled = scaler.transform(X)

    final_model = MultiOutputRegressor(RandomForestRegressor(
        n_estimators=1000, max_depth=2, min_samples_split=2, min_samples_leaf=3, random_state=0
    ))
    final_model.fit(X_scaled, y)

    with open(f"{models_pickle_path}best_model_{arquive}.pickle", 'wb') as f:
        pickle.dump(best_model, f)
    with open(f"{models_pickle_path}worst_model_{arquive}.pickle", 'wb') as f:
        pickle.dump(worst_model, f)
    with open(f"{models_pickle_path}final_model_{arquive}.pickle", 'wb') as f:
        pickle.dump(final_model, f)

    instances = df.iloc[:, 0]
    as_df = pd.DataFrame({'Instance': instances[:len(as_predictions_raw[0])],
                          'AS_Prediction': np.mean(as_predictions_raw, axis=0)})
    vbs_df = pd.DataFrame({'Instance': instances[:len(vbs_predictions_raw[0])],
                           'VBS_Prediction': np.mean(vbs_predictions_raw, axis=0)})
    sbs_df = pd.DataFrame({'Instance': instances[:len(sbs_predictions_raw[0])],
                           'SBS_Prediction': np.mean(sbs_predictions_raw, axis=0)})

    os.makedirs(f"{models_merit_path}AS/", exist_ok=True)
    os.makedirs(f"{models_merit_path}VBS/", exist_ok=True)
    os.makedirs(f"{models_merit_path}SBS/", exist_ok=True)

    as_df.to_csv(f"{models_merit_path}AS/{arquive}_ASModel.csv", index=False)
    vbs_df.to_csv(f"{models_merit_path}VBS/{arquive}_VBSModel.csv", index=False)
    sbs_df.to_csv(f"{models_merit_path}SBS/{arquive}_SBSModel.csv", index=False)

    pd.DataFrame({'Fold': fold_numbers, 'SBS_Best_Models': sbs_best_model}) \
        .to_csv(f"{models_merit_path}SBS/{arquive}_SBS_chosen_algorithms.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", type=str)
    parser.add_argument("arquive", type=str)
    parser.add_argument("label", type=str)
    args = parser.parse_args()
    
    models_and_merit_builder(args.file_path, args.arquive, args.label)
