import os
import pandas as pd
import numpy as np
import sys
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def _load_pair(vbs_path, as_path):
    if not os.path.isfile(vbs_path):
        raise FileNotFoundError(f"Missing VBS file: {vbs_path}")
    if not os.path.isfile(as_path):
        raise FileNotFoundError(f"Missing AS file: {as_path}")
    vbs_df = pd.read_csv(vbs_path)
    as_df = pd.read_csv(as_path)
    required_v = {"Instance", "VBS_Prediction"}
    required_a = {"Instance", "AS_Prediction"}
    if not required_v.issubset(vbs_df.columns):
        raise ValueError(f"VBS schema invalid: {vbs_path}")
    if not required_a.issubset(as_df.columns):
        raise ValueError(f"AS schema invalid: {as_path}")
    merged = pd.merge(vbs_df, as_df, on="Instance")
    merged["VBS_Prediction"] = pd.to_numeric(merged["VBS_Prediction"], errors='coerce')
    merged["AS_Prediction"] = pd.to_numeric(merged["AS_Prediction"], errors='coerce')
    merged = merged.dropna(subset=["VBS_Prediction", "AS_Prediction"])
    if merged.empty:
        raise ValueError("Merged dataframe empty after coercion.")
    return merged

def generate_regression_metric_tables(label: str):
    """Gera e salva as tabelas de métricas de regressão para um determinado *label*.

    Retorna um dicionário com os DataFrames para reutilização programática.
    """
    models_path = f'result/{label}/models/theoretical_models/'
    output_path = f'result/{label}/regression_metrics/'
    os.makedirs(output_path, exist_ok=True)

    l_values = [100, 50, 25, 10, 5]
    r_values = ["1.0", "0.5", "0.25", "0.1", "0.05"]

    mse_table = pd.DataFrame(index=r_values, columns=l_values)
    rmse_table = pd.DataFrame(index=r_values, columns=l_values)
    mae_table = pd.DataFrame(index=r_values, columns=l_values)
    mape_table = pd.DataFrame(index=r_values, columns=l_values)
    r2_table = pd.DataFrame(index=r_values, columns=l_values)

    print("-> Getting regression metrics for random walk")
    for l in l_values:
        for r in r_values:
            vbs_path = os.path.join(models_path, f"l{l}_r{r}", "VBS", f"l{l}_r{r}_VBSModel.csv").replace("\\", "/")
            as_path = os.path.join(models_path, f"l{l}_r{r}", "AS", f"l{l}_r{r}_ASModel.csv").replace("\\", "/")
            try:
                merged_df = _load_pair(vbs_path, as_path)
                mse_value = mean_squared_error(merged_df["VBS_Prediction"], merged_df["AS_Prediction"])
                rmse_value = np.sqrt(mse_value)
                mae_value = mean_absolute_error(merged_df["VBS_Prediction"], merged_df["AS_Prediction"])
                valid_indices = merged_df["VBS_Prediction"] != 0
                mape_value = (np.abs((merged_df["VBS_Prediction"] - merged_df["AS_Prediction"]) / merged_df["VBS_Prediction"]))[valid_indices].mean() * 100
                r2_value = r2_score(merged_df["VBS_Prediction"], merged_df["AS_Prediction"])
                mse_table.at[r, l] = round(mse_value, 10)
                rmse_table.at[r, l] = round(rmse_value, 10)
                mae_table.at[r, l] = round(mae_value, 10)
                mape_table.at[r, l] = round(mape_value, 10) if not np.isnan(mape_value) else "Erro"
                r2_table.at[r, l] = round(r2_value, 10)
            except Exception as e:
                print(f"Error processing l={l}, r={r}: {e}")
                mse_table.at[r, l] = "Erro"
                rmse_table.at[r, l] = "Erro"
                mae_table.at[r, l] = "Erro"
                mape_table.at[r, l] = "Erro"
                r2_table.at[r, l] = "Erro"

    print("-> Getting regression metrics for adaptive walk")
    adaptive_walk_mse_values = []
    adaptive_walk_rmse_values = []
    adaptive_walk_mae_values = []
    adaptive_walk_mape_values = []
    adaptive_walk_r2_values = []

    for r in r_values:
        vbs_path = os.path.join(models_path, f"r{r}", "VBS", f"r{r}_VBSModel.csv").replace("\\", "/")
        as_path = os.path.join(models_path, f"r{r}", "AS", f"r{r}_ASModel.csv").replace("\\", "/")
        try:
            vbs_df = _load_pair(vbs_path, as_path)
            mse_value = mean_squared_error(vbs_df["VBS_Prediction"], vbs_df["AS_Prediction"])
            rmse_value = np.sqrt(mse_value)
            mae_value = mean_absolute_error(vbs_df["VBS_Prediction"], vbs_df["AS_Prediction"])
            valid_indices = vbs_df["VBS_Prediction"] != 0
            mape_value = (np.abs((vbs_df["VBS_Prediction"] - vbs_df["AS_Prediction"]) / vbs_df["VBS_Prediction"]))[valid_indices].mean() * 100
            r2_value = r2_score(vbs_df["VBS_Prediction"], vbs_df["AS_Prediction"])
            adaptive_walk_mse_values.append(round(mse_value, 10))
            adaptive_walk_rmse_values.append(round(rmse_value, 10))
            adaptive_walk_mae_values.append(round(mae_value, 10))
            adaptive_walk_mape_values.append(round(mape_value, 10) if not np.isnan(mape_value) else "Erro")
            adaptive_walk_r2_values.append(round(r2_value, 10))
        except Exception as e:
            print(f"Error processing r={r}: {e}")
            adaptive_walk_mse_values.append("Erro")
            adaptive_walk_rmse_values.append("Erro")
            adaptive_walk_mae_values.append("Erro")
            adaptive_walk_mape_values.append("Erro")
            adaptive_walk_r2_values.append("Erro")

    mse_table["Adaptive Walk"] = adaptive_walk_mse_values
    rmse_table["Adaptive Walk"] = adaptive_walk_rmse_values
    mae_table["Adaptive Walk"] = adaptive_walk_mae_values
    mape_table["Adaptive Walk"] = adaptive_walk_mape_values
    r2_table["Adaptive Walk"] = adaptive_walk_r2_values

    for table in [mse_table, rmse_table, mae_table, mape_table, r2_table]:
        table.index = [f"{int(float(r) * 100)}%" for r in r_values]
        table.columns = [f"ℓ = {l}" for l in l_values] + ["Adaptive Walk"]
        table.index.name = "r"

    mse_table.to_csv(os.path.join(output_path, "mse.csv"), float_format="%.10f")
    rmse_table.to_csv(os.path.join(output_path, "rmse.csv"), float_format="%.10f")
    mae_table.to_csv(os.path.join(output_path, "mae.csv"), float_format="%.10f")
    mape_table.to_csv(os.path.join(output_path, "mape.csv"), float_format="%.10f")
    r2_table.to_csv(os.path.join(output_path, "r2.csv"), float_format="%.10f")

    return {
        'mse': mse_table,
        'rmse': rmse_table,
        'mae': mae_table,
        'mape': mape_table,
        'r2': r2_table,
        'output_path': output_path,
    }


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Uso: python regression_metrics.py <label>")
        sys.exit(1)
    label_arg = sys.argv[1]
    generate_regression_metric_tables(label_arg)

