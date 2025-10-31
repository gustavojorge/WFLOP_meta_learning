import argparse
import json
import os
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

from search.random_search import run_random_search
from search.grid_search import run_grid_search
from evaluation.merit import compute_merit_for_model
from utils.io import ensure_dir, save_json

def build_y_classes(df, label):
    """
    Constrói vetor de classes (y_classes) baseado no melhor algoritmo por instância,
    conforme o label (epsilon, hipervolume ou igd).
    """
    # Import dinâmico dos índices de algoritmos (como no seu pipeline de mérito)
    from components.build_indices_dict import build_all_indices

    all_indices = build_all_indices()
    if label not in all_indices:
        raise ValueError(f"Label '{label}' não encontrado em índices dinâmicos")

    label_indices = all_indices[label]
    nsga_indices = set(label_indices["nsga"])
    moead_indices = set(label_indices["moead"])
    comolsd_indices = set(label_indices["comolsd"])

    y_classes = []
    for inst_id in df.iloc[:, 0].astype(str):
        if inst_id in nsga_indices:
            y_classes.append(0)
        elif inst_id in moead_indices:
            y_classes.append(1)
        elif inst_id in comolsd_indices:
            y_classes.append(2)
        else:
            raise ValueError(f"Instância {inst_id} não encontrada em nenhum grupo de índices.")

    return pd.Series(y_classes, name="y_classes")

def main():
    parser = argparse.ArgumentParser(description='Param search orchestrator (Random -> Grid) minimizing merit m.')
    parser.add_argument('--label', required=True, choices=['epsilon', 'hipervolume', 'igd'])
    parser.add_argument('--l', required=True, type=int)
    parser.add_argument('--r', required=True, type=str)
    parser.add_argument('--random-iters', type=int, default=20)
    parser.add_argument('--folds-merit', type=int, default=30)
    parser.add_argument('--cv', type=int, default=3)
    parser.add_argument('--n-jobs', type=int, default=-1)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    dataset_path = f'meta_dataset/{args.label}/pareto_based/random_walk/l{args.l}_r{args.r}.csv'
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f'Dataset not found: {dataset_path}')
    df = pd.read_csv(dataset_path)

    algo_cols = df.columns[-3:]
    feature_cols = df.columns[1:-3]
    X, y = df[feature_cols], df[algo_cols]

    # === Construção do vetor de classes (para SMOTE) ===
    print("Construindo vetor de classes (y_classes)...")
    y_classes = build_y_classes(df, args.label).values

    base_model = MultiOutputRegressor(RandomForestRegressor(random_state=0))

    print("=== Random Search Phase ===")
    random_best_model, best_random_params = run_random_search(base_model, X, y, args, y_classes)
    # merit_random = compute_merit_for_model(random_best_model, df, args.label, n_folds=args.folds_merit)
    # print("Merit after Random Search:", merit_random)

    """ print("=== Grid Search Phase ===")
    grid_best_model, best_grid_params = run_grid_search(base_model, X, y, best_random_params, args, y_classes)
    merit_grid = compute_merit_for_model(grid_best_model, df, args.label, n_folds=args.folds_merit)
    print("Merit after Grid Search:", merit_grid)"""

    results = {
        'dataset': dataset_path,
        'label': args.label,
        'l': args.l,
        'r': args.r,
        'random_search': {'best_params': best_random_params},
    }

    output_path = args.output or f'params_model/{args.label}.json'
    ensure_dir(Path(output_path).parent)
    save_json(results, output_path)
    print(f"Results saved to {output_path}")


if __name__ == '__main__':
    main()
