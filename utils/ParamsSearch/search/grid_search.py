import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from imblearn.over_sampling import SMOTE
from sklearn.base import clone


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


def normalized_mae(y_true, y_pred):
    """
    NMAE = MAE / (max(y_true) - min(y_true))
    Calculado para cada saída e depois feito o mean across targets.
    """
    if isinstance(y_true, pd.DataFrame) or isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.DataFrame) or isinstance(y_pred, pd.Series):
        y_pred = y_pred.values

    n_targets = y_true.shape[1] if y_true.ndim > 1 else 1
    nmae_values = []
    for i in range(n_targets):
        yt = y_true[:, i] if n_targets > 1 else y_true
        yp = y_pred[:, i] if n_targets > 1 else y_pred
        denom = np.max(yt) - np.min(yt)
        if denom == 0:
            nmae_values.append(0.0)
        else:
            mae = mean_absolute_error(yt, yp)
            nmae_values.append(mae / denom)
    return np.mean(nmae_values)


def run_grid_search(base_model, X, y, best_random_params, args, y_classes):
    """
    Executa Grid Search manualmente com SMOTE dentro dos folds (sem vazamento),
    usando NMAE (Normalized Mean Absolute Error) como métrica.
    """
    print("=== Grid Search com SMOTE dentro dos folds (sem vazamento, NMAE) ===")

    grid_params = derive_grid_from_random(best_random_params)
    grid_combinations = list(ParameterGrid(grid_params))

    kf = KFold(n_splits=args.cv, shuffle=True, random_state=0)
    best_score = float('inf')
    best_params = None
    best_model = None

    for i, params in enumerate(grid_combinations):
        print(f"\n--- Avaliando grid {i+1}/{len(grid_combinations)}: {params} ---")
        fold_scores = []

        for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            y_classes_train = y_classes[train_idx]

            # === SMOTE aplicado apenas no treino ===
            unique, counts = np.unique(y_classes_train, return_counts=True)
            min_count = counts.min()
            if min_count < 3:
                print(f"[Fold {fold+1}] Poucas instâncias em alguma classe (<3). SMOTE desativado.")
                X_res, y_res = X_train, y_train
            else:
                k_neighbors = max(1, min(min_count - 1, 5))
                smote = SMOTE(random_state=0, k_neighbors=k_neighbors)
                combined_train = X_train.reset_index(drop=True)
                X_res, y_classes_res = smote.fit_resample(combined_train, y_classes_train)

                # Ajusta o número de y_res para combinar com X_res (mantendo targets numéricos)
                synth_count = X_res.shape[0] - X_train.shape[0]
                y_synth = y_train.sample(n=synth_count, replace=True, random_state=0).reset_index(drop=True)
                y_res = pd.concat([y_train.reset_index(drop=True), y_synth], ignore_index=True)

            # === Normalização ===
            scaler = StandardScaler()
            X_res_scaled = scaler.fit_transform(X_res)
            X_test_scaled = scaler.transform(X_test)

            # === Treinamento e avaliação ===
            model = clone(base_model)
            model.set_params(**params)
            model.fit(X_res_scaled, y_res)

            y_pred = model.predict(X_test_scaled)
            score = normalized_mae(y_test, y_pred)
            fold_scores.append(score)

        mean_score = np.mean(fold_scores)
        print(f"Média NMAE (cross-val): {mean_score:.6f}")

        # Atualiza melhor combinação
        if mean_score < best_score:
            best_score = mean_score
            best_params = params
            best_model = clone(base_model).set_params(**params)
            best_model.fit(StandardScaler().fit_transform(X), y)

    print("\n=== Melhor combinação encontrada ===")
    print(best_params)
    print(f"Melhor NMAE médio: {best_score:.6f}")

    return best_model, best_params
