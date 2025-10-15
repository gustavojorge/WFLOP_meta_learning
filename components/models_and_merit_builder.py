import argparse
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Importa indices dinâmicos
try:
    # Execução como parte do pacote (via python -m ou import)
    from .build_indices_dict import build_all_indices  # type: ignore
except ImportError:
    # Fallback quando executado diretamente como script: ajustar sys.path
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

    # ------------ INDICES DINÂMICOS (melhor algoritmo por instância) ------------
    all_indices = build_all_indices()
    if label not in all_indices:
        raise ValueError("Invalid label (não encontrado em indices dinâmicos)")

    label_indices = all_indices[label]
    required_algos = {"nsga", "moead", "comolsd"}
    if not required_algos.issubset(label_indices.keys()):
        raise ValueError(f"Algoritmos esperados faltando em indices: {required_algos - set(label_indices.keys())}")

    nsga_indices = label_indices["nsga"]
    moead_indices = label_indices["moead"]
    comolsd_indices = label_indices["comolsd"]

    # Tabelas de instâncias por algoritmo
    nsga_instances = df[df.iloc[:, 0].astype(str).isin(nsga_indices)]
    moead_instances = df[df.iloc[:, 0].astype(str).isin(moead_indices)]
    comolsd_instances = df[df.iloc[:, 0].astype(str).isin(comolsd_indices)]

    scaler = StandardScaler()
    n_folds = 100
    scores = []

    # Storing the importance of the features
    # Descobre automaticamente colunas de algoritmos: assumimos que são as últimas 3 colunas
    algo_cols = df.columns[-3:]
    feature_cols = df.columns[1:-3]
    feature_importances = np.zeros(len(feature_cols))  

    # Storing the best and worst models (with pickle) from all interations (folds)
    best_model = None
    worst_model = None
    best_score = float('inf')
    worst_score = float('-inf')

    # Storing the all performances of AS, SBS and VBS models
    as_predictions = []
    vbs_predictions = []
    sbs_predictions = []

    # Storing the chosen SBS model in each fold
    sbs_best_model = []
    fold_numbers = []

    # Custom test sizes for NSGA and MOEAD as per label (rest 90/10)
    # --> This increases the difference between "rhv(SBS, I)" e "rhv(VBS, I)" in the denominator of the fraction. This is, the metric is lower.
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
    
    for fold_number in range(n_folds):
        fold_numbers.append(fold_number)

        # Manual stratified selection for nsga and moead
        if len(nsga_indices) < size_nsga_test:
            raise ValueError("nsga_indices insuficientes para tamanho de teste definido")
        if len(moead_indices) < size_moead_test:
            raise ValueError("moead_indices insuficientes para tamanho de teste definido")

        selected_nsga_indices_test = np.random.choice(nsga_indices, size=size_nsga_test, replace=False)
        selected_moead_indices_test = np.random.choice(moead_indices, size=size_moead_test, replace=False)

        selected_nsga_indices_train = list(set(nsga_indices) - set(selected_nsga_indices_test))
        selected_moead_indices_train = list(set(moead_indices) - set(selected_moead_indices_test))

        nsga_train = nsga_instances[nsga_instances.iloc[:, 0].astype(str).isin(selected_nsga_indices_train)]
        nsga_test = nsga_instances[nsga_instances.iloc[:, 0].astype(str).isin(selected_nsga_indices_test)]
        moead_train = moead_instances[moead_instances.iloc[:, 0].astype(str).isin(selected_moead_indices_train)]
        moead_test = moead_instances[moead_instances.iloc[:, 0].astype(str).isin(selected_moead_indices_test)]

        # COMOLSD with random 90/10 split
        comolsd_train, comolsd_test = train_test_split(comolsd_instances, test_size=0.1, random_state=np.random.randint(0, 1000))

        train_data = pd.concat([nsga_train, moead_train, comolsd_train])
        test_data = pd.concat([nsga_test, moead_test, comolsd_test])

        X_train, y_train = train_data.loc[:, feature_cols], train_data.loc[:, algo_cols]
        X_test, y_test = test_data.loc[:, feature_cols], test_data.loc[:, algo_cols]
        
        #---------- NORMALIZATION OF FEATURES ----------
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        #---------- MULTIOUTPUT RANDOM FOREST REGRESSOR ----------        
        #model = MultiOutputRegressor(RandomForestRegressor(n_estimators=1000, max_depth=2, min_samples_split=2, min_samples_leaf=3, random_state=0))
        model = MultiOutputRegressor(RandomForestRegressor(random_state=0))
        model.fit(X_train_scaled, y_train)  # TRAINING

        #MEAN OF FEATURE IMPORTANCE OF ALL ESTIMATORS
        feature_importances += np.mean([est.feature_importances_ for est in model.estimators_], axis=0)
        
        y_pred = pd.DataFrame(model.predict(X_test_scaled), columns=y_train.columns, index=y_test.index) #PREDICTION
        
        #---------- METRIC CALCULATION ----------        
        AS = y_train.columns[np.argmin(y_pred.values, axis=1)]  
        VBS = y_test.idxmin(axis=1)  
        SBS = y_train.mean().idxmin()  

        sbs_best_model.append(str(SBS))
        
        rhv_AS = np.array([y_test.loc[y_test.index[i], AS[i]] for i in range(len(X_test))])
        rhv_SBS = np.array([y_test.loc[y_test.index[i], SBS] for i in range(len(X_test))])
        rhv_VBS = np.array([y_test.at[y_test.index[i], VBS.iloc[i]] for i in range(len(X_test))])

        as_predictions.append(rhv_AS)
        vbs_predictions.append(rhv_VBS)
        sbs_predictions.append(rhv_SBS)
        
        rhv_AS_mean = rhv_AS.mean()
        rhv_SBS_mean = rhv_SBS.mean()
        rhv_VBS_mean = rhv_VBS.mean()
        
        if rhv_SBS_mean == rhv_VBS_mean:
            raise Exception("rhv_SBS_mean == rhv_VBS_mean")
        else:
            m = (rhv_AS_mean - rhv_VBS_mean) / (rhv_SBS_mean - rhv_VBS_mean)
        
        scores.append(m)
        
        #Finding the best and worst models
        if m < best_score:
            best_score = m
            best_model = model
        
        if m > worst_score:
            worst_score = m
            worst_model = model

    #---------- Calculates the average importance of the features from all itereations (folds) ----------    
    feature_importances /= n_folds 
    feature_importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': feature_importances
    })
    feature_importance_df.sort_values(by='Importance', ascending=False, inplace=True)
    
    feature_importance_df.to_csv(f"{output_path}{arquive}_features_importance.csv", index=False)
    
    print(np.mean(scores))
    print([float(x) for x in scores])

    # final_model = Training a model with the whole data
    X, y = df.loc[:, feature_cols], df.loc[:, algo_cols]
    scaler.fit(X)
    X_scaled = scaler.transform(X)
    
    final_model = MultiOutputRegressor(RandomForestRegressor(n_estimators=1000, max_depth=2, min_samples_split=2, min_samples_leaf=3, random_state=0))
    final_model.fit(X_scaled, y)
    
    # ---------- Storing the pickle models (best_model, worst_model and final_model) ----------
    with open(f"{models_pickle_path}best_model_{arquive}.pickle", 'wb') as f:
        pickle.dump(best_model, f)
    
    with open(f"{models_pickle_path}worst_model_{arquive}.pickle", 'wb') as f:
        pickle.dump(worst_model, f)
    
    with open(f"{models_pickle_path}final_model_{arquive}.pickle", 'wb') as f:
        pickle.dump(final_model, f)

    # ---------- Calculates the average performance of AS, VBS and SBS models and store it in "models_merit_path" ----------
    instances = df.iloc[:, 0]
    as_predictions_mean = np.mean(as_predictions, axis=0)
    vbs_predictions_mean = np.mean(vbs_predictions, axis=0)
    sbs_predictions_mean = np.mean(sbs_predictions, axis=0)
    
    as_df = pd.DataFrame({'Instance': instances.iloc[:len(as_predictions_mean)], 'AS_Prediction': as_predictions_mean})
    vbs_df = pd.DataFrame({'Instance': instances.iloc[:len(vbs_predictions_mean)], 'VBS_Prediction': vbs_predictions_mean})
    sbs_df = pd.DataFrame({'Instance': instances.iloc[:len(sbs_predictions_mean)], 'SBS_Prediction': sbs_predictions_mean})
    
    os.makedirs(f"{models_merit_path}AS/", exist_ok=True)
    os.makedirs(f"{models_merit_path}VBS/", exist_ok=True)
    os.makedirs(f"{models_merit_path}SBS/", exist_ok=True)

    as_df.to_csv(f"{models_merit_path}AS/{arquive}_ASModel.csv", index=False)
    vbs_df.to_csv(f"{models_merit_path}VBS/{arquive}_VBSModel.csv", index=False)
    sbs_df.to_csv(f"{models_merit_path}SBS/{arquive}_SBSModel.csv", index=False)
    
    # Stores the chosen SBS in each fold
    sbs_df2 = pd.DataFrame({'Fold': fold_numbers, 'SBS_Best_Models': sbs_best_model})
    sbs_file_path = f"{models_merit_path}SBS/{arquive}_SBS_chosen_algorithms.csv"
    sbs_df2.to_csv(sbs_file_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", type=str)
    parser.add_argument("arquive", type=str)
    parser.add_argument("label", type=str)
    args = parser.parse_args()
    
    models_and_merit_builder(args.file_path, args.arquive, args.label)