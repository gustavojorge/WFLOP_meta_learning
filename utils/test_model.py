import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

def test_best_model(arquive):
    file_path = f'final_dataset/pareto_based/random_walk/{arquive}.csv'
    df = pd.read_csv(file_path)

    test_instance = df.iloc[0, 1:-2].values.reshape(1, -1)  

    model_path = f'refined_results/meta_learning/epsilon/models_pickle/{arquive}/best_model_fold.pickle'
    with open(model_path, 'rb') as f:
        best_model = pickle.load(f)

    scaler = StandardScaler()
    scaler.fit(df.iloc[:, 1:-2])  
    test_instance_scaled = scaler.transform(test_instance)

    y_pred = best_model.predict(test_instance_scaled)
    print(y_pred)

if __name__ == "__main__":
    arquive = 'l5_r0.1'
    test_best_model(arquive)
