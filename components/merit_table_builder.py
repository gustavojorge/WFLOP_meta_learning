import os
import sys
import time
import pandas as pd
import subprocess

def run_merit_script(script_path, dataset_path, arquive, label, log_path, python_exec):
    os.makedirs(log_path, exist_ok=True)
    try:
        start_time = time.time()
        result = subprocess.run(
            [python_exec, script_path, dataset_path, arquive, label],
            capture_output=True, text=True, check=True
        )
        end_time = time.time()
        run_time = end_time - start_time
        
        output_lines = [ln.strip() for ln in result.stdout.strip().split("\n") if ln.strip()]
        # Estratégia robusta: encontra a primeira linha que converte para float (média) e
        # depois uma linha com representação de lista [ ... ] para méritos por fold.
        metric_value = None
        merit_list = []
        for ln in output_lines:
            if metric_value is None:
                try:
                    metric_value = float(ln)
                    continue
                except ValueError:
                    # não é valor numérico da média
                    pass
            # Detecta lista de méritos
            if ln.startswith("[") and ln.endswith("]"):
                # remove colchetes e separa por vírgula
                inner = ln[1:-1].strip()
                if inner:
                    merit_list = [x.strip() for x in inner.split(",")]
                break
        if metric_value is None:
            raise ValueError(f"Não foi possível extrair métrica numérica do stdout. Linhas: {output_lines[:5]}")
        
        log_file_path = os.path.join(log_path, f"{arquive}_log.txt")
        with open(log_file_path, 'w', encoding='utf-8') as log_file:
            log_file.write("================== LOG ==================\n")
            log_file.write(f"Dataset: {dataset_path}\n")
            log_file.write(f"Parameters: {arquive}\n")
            log_file.write(f"Label: {label}\n")
            log_file.write(f"Run time: {run_time:.4f} seconds\n")
            log_file.write(f"Merit Average: {metric_value:.4f}\n")
            log_file.write(f"Merit List: {', '.join(merit_list)}\n")
            if result.stderr.strip():
                log_file.write("--- STDERR ---\n")
                log_file.write(result.stderr)
        
        return round(metric_value, 3)
    except Exception as e:
        log_file_path = os.path.join(log_path, f"{arquive}_log.txt")
        with open(log_file_path, 'w', encoding='utf-8') as log_file:
            log_file.write("================== ERROR LOG ==================\n")
            log_file.write(f"Dataset: {dataset_path}\n")
            log_file.write(f"Parameters: {arquive}\n")
            log_file.write(f"Label: {label}\n")
            log_file.write(f"Exception: {repr(e)}\n")
        print(f"Error processing {arquive}: {e}")
        return "Erro"

def process_random_walk(label, script_path, base_path, l_values, r_values, results_table, python_exec):
    print("-> Building the dataset for random walk")
    log_path = f"result/{label}/logs/"
    
    for l in l_values:
        for r in r_values:
            print(f"--> Building for r = {r} and l = {l}")
            dataset_path = os.path.join(base_path, f"l{l}_r{r}.csv")
            results_table.at[r, l] = run_merit_script(script_path, dataset_path, f"l{l}_r{r}", label, log_path, python_exec)

def process_adaptive_walk(label, script_path, base_path, r_values, python_exec):
    print("\n-> Building the dataset for adaptive walk")
    log_path = f"result/{label}/logs/"
    adaptive_walk_values = []
    
    for r in r_values:
        print(f"--> Building for r = {r}")
        dataset_path = os.path.join(base_path, f"r{r}.csv")
        adaptive_walk_values.append(run_merit_script(script_path, dataset_path, f"r{r}", label, log_path, python_exec))
    
    return adaptive_walk_values

def main():
    if len(sys.argv) != 2:
        print("Uso: python builder_merit_table.py <label>")
        sys.exit(1)
    
    label = sys.argv[1]

    # Atualizado para nova estrutura de diretórios
    base_path_random = f'meta_dataset/{label}/pareto_based/random_walk/'
    base_path_adaptive = f'meta_dataset/{label}/pareto_based/adaptive_walk/'
    output_path = f'result/{label}/'
    script_path = 'components/models_and_merit_builder.py'
    python_exec = sys.executable
    
    l_values = [100, 50, 25, 10, 5]
    r_values = ["1.0", "0.5", "0.25", "0.1", "0.05"]

    results_table = pd.DataFrame(index=r_values, columns=l_values)
    
    os.makedirs(output_path, exist_ok=True)

    process_random_walk(label, script_path, base_path_random, l_values, r_values, results_table, python_exec) 
    adaptive_walk_values = process_adaptive_walk(label, script_path, base_path_adaptive, r_values, python_exec)

    # Validação de comprimento: deve ter um valor por r
    if len(adaptive_walk_values) != len(r_values):
        print(f"[WARN] adaptive_walk_values length={len(adaptive_walk_values)} != expected {len(r_values)}. Padding com 'Erro'.")
        # Preenche até tamanho correto
        while len(adaptive_walk_values) < len(r_values):
            adaptive_walk_values.append('Erro')
        if len(adaptive_walk_values) > len(r_values):
            adaptive_walk_values = adaptive_walk_values[:len(r_values)]

    results_table["Adaptive Walk"] = adaptive_walk_values
    results_table.index = [f"{int(float(r) * 100)}%" for r in r_values]
    results_table.columns = [f"\u2113 = {l}" for l in l_values] + ["Adaptive Walk"]
    results_table.index.name = "r"
    
    output_csv = os.path.join(output_path, f"merit_{label}.csv")
    # Substitui strings 'Erro' por NaN para facilitar análise posterior
    results_table.replace('Erro', pd.NA, inplace=True)
    results_table.to_csv(output_csv)    
    
    print("\nDONE! Table with merit values generated and saved")

if __name__ == "__main__":
    main()
