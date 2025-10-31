import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, ParameterSampler
from scipy.stats import randint
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from imblearn.over_sampling import SMOTE, RandomOverSampler
from collections import Counter

# ============================================================
# === Funções auxiliares =====================================
# ============================================================

def instance_minmax(df_algos: pd.DataFrame) -> pd.DataFrame:
    """Normalização por instância (linha) min-max nas colunas de algoritmos."""
    row_min = df_algos.min(axis=1)
    row_max = df_algos.max(axis=1)
    span = row_max - row_min
    span = span.replace(0, 1)
    return (df_algos.sub(row_min, axis=0)).div(span, axis=0)

def merit_metric(y_true, y_pred):
    """
    Calcula o mérito médio (m) como no pipeline de mérito.
    Pula instâncias em que SBS == VBS (denominador nulo).
    """
    y_true = pd.DataFrame(y_true)
    y_pred = pd.DataFrame(y_pred, columns=y_true.columns)

    AS = y_pred.columns[np.argmin(y_pred.values, axis=1)]
    VBS = y_true.idxmin(axis=1)
    SBS = y_true.mean().idxmin()

    merits = []
    for i in range(len(y_true)):
        rhv_AS = y_true.loc[y_true.index[i], AS[i]]
        rhv_SBS = y_true.loc[y_true.index[i], SBS]
        rhv_VBS = y_true.loc[y_true.index[i], VBS.iloc[i]]

        if rhv_SBS == rhv_VBS:
            continue

        merits.append((rhv_AS - rhv_VBS) / (rhv_SBS - rhv_VBS))

    if not merits:
        print("Warning: all instances skipped in this fold merit calculation.")
        return np.nan

    return np.mean(merits)

# ============================================================
# === Random Search com SMOTE e Mérito ========================
# ============================================================

def run_random_search(base_model, X, y, args, y_classes):
    """
    Executa Random Search manualmente com SMOTE moderado dentro dos folds (sem vazamento)
    e otimiza diretamente o mérito médio (m).
    """

    print("=== Random Search com SMOTE moderado e métrica de mérito ===")
    print("Distribuição original de classes:", Counter(y_classes))

    # Espaço de parâmetros
    param_dist = {
        'estimator__n_estimators': randint(100, 1000),
        'estimator__max_depth': [None, 2, 4, 10, 20, 25],
        'estimator__min_samples_split': randint(2, 10),
        'estimator__min_samples_leaf': randint(1, 5)
    }

    param_samples = list(ParameterSampler(param_dist, n_iter=args.random_iters, random_state=0))
    kf = KFold(n_splits=args.cv, shuffle=True, random_state=0)

    best_score = float('inf')
    best_params = None
    best_model = None

    for i, params in enumerate(param_samples):
        print(f"\n--- Avaliando combinação {i+1}/{len(param_samples)}: {params} ---")
        fold_merits = []

        for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            y_classes_train = y_classes[train_idx]

            # === Aplica SMOTE moderado no treino ===
            unique, counts = np.unique(y_classes_train, return_counts=True)
            max_count = counts.max()
            target_minor = int(max_count * 0.85)
            sampling_strategy = {cls: min(max_count, target_minor if c < target_minor else c)
                                 for cls, c in zip(unique, counts)}

            if counts.min() < 3:
                print(f"[Fold {fold+1}] Poucas instâncias em alguma classe (<3). Usando RandomOverSampler.")
                ros = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=0)
                X_res, y_classes_res = ros.fit_resample(X_train.reset_index(drop=True), y_classes_train)
            else:
                k_neighbors = max(1, min(counts.min() - 1, 5))
                smote = SMOTE(sampling_strategy=sampling_strategy, k_neighbors=k_neighbors, random_state=0)
                X_res, y_classes_res = smote.fit_resample(X_train.reset_index(drop=True), y_classes_train)

            synth_count = X_res.shape[0] - X_train.shape[0]
            y_synth = y_train.sample(n=synth_count, replace=True, random_state=0).reset_index(drop=True)
            y_res = pd.concat([y_train.reset_index(drop=True), y_synth], ignore_index=True)

            print(f"[Fold {fold+1}] Distribuição balanceada: {Counter(y_classes_res)}")

            # === Normalização por instância (y) ===
            y_res_norm = instance_minmax(y_res)
            y_test_norm = instance_minmax(y_test)

            # === Escala de features ===
            scaler = StandardScaler()
            X_res_scaled = scaler.fit_transform(X_res)
            X_test_scaled = scaler.transform(X_test)

            # === Treina modelo ===
            model = clone(base_model)
            model.set_params(**params)
            model.fit(X_res_scaled, y_res_norm)

            # === Avalia mérito ===
            y_pred = model.predict(X_test_scaled)
            score = merit_metric(y_test_norm, y_pred)
            fold_merits.append(score)

        mean_merit = np.mean(fold_merits)
        print(f"Média mérito (cross-val): {mean_merit:.6f}")

        # === Atualiza melhor combinação ===
        if mean_merit < best_score:
            best_score = mean_merit
            best_params = params
            best_model = clone(base_model).set_params(**params)
            X_scaled_full = StandardScaler().fit_transform(X)
            y_norm_full = instance_minmax(y)
            best_model.fit(X_scaled_full, y_norm_full)

    print("\n=== Melhor combinação encontrada ===")
    print(best_params)
    print(f"Melhor mérito médio: {best_score:.6f}")

    return best_model, best_params
