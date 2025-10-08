import os
import json
import argparse
import pandas as pd
from typing import Dict, List, Optional


INDICATORS = ["epsilon", "hipervolume", "igd"]  # manter chave 'hipervolume' consistente com o código existente
ALGO_COLUMNS = ["MOEAD", "COMOLSD", "NSGAII"]


def _locate_indicator_csv(indicator: str, preferred_filename: str = "r1.0.csv") -> Optional[str]:
    """Localiza um CSV representativo para o indicador.

    Ordem de busca:
        1. meta_dataset/<indicator>/pareto_based/adaptive_walk/r1.0.csv (ou preferred_filename)
        2. Primeiro *.csv dentro de meta_dataset/<indicator>/pareto_based/adaptive_walk/
    Retorna caminho absoluto relativo ao projeto ou None se não encontrar.
    """
    base_dir = os.path.join("meta_dataset", indicator, "pareto_based", "adaptive_walk")
    preferred_path = os.path.join(base_dir, preferred_filename)
    if os.path.isfile(preferred_path):
        return preferred_path
    if os.path.isdir(base_dir):
        for fname in sorted(os.listdir(base_dir)):
            if fname.endswith('.csv'):
                return os.path.join(base_dir, fname)
    return None


def _normalize_algo_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Garante que as colunas de algoritmos existam e estejam com nomes padronizados.
    Aceita variações de caixa (case-insensitive)."""
    col_map = {c.lower(): c for c in df.columns}
    renamed = {}
    for target in ALGO_COLUMNS:
        lower = target.lower()
        if lower in col_map and col_map[lower] != target:
            renamed[col_map[lower]] = target
    if renamed:
        df = df.rename(columns=renamed)
    # Verificação final
    missing = [c for c in ALGO_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas de algoritmos ausentes no CSV: {missing}")
    return df


def build_indices_for_indicator(indicator: str, tie_strategy: str = "all") -> Dict[str, List[str]]:
    """Constroi o dicionário de listas (moead/nsga/comolsd) para um indicador.

    tie_strategy:
        - 'all': em caso de empate inclui a instância em todas as listas empatadas.
        - 'first': inclui somente a primeira (ordem ALGO_COLUMNS).
    Critério: menor valor => melhor (mesma regra solicitada para todos os indicadores).
    """
    csv_path = _locate_indicator_csv(indicator)
    if not csv_path:
        raise FileNotFoundError(f"Nenhum CSV encontrado para indicador '{indicator}'.")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"CSV '{csv_path}' está vazio.")

    df = _normalize_algo_columns(df)

    # Assume primeira coluna como identificador da instância
    instance_col = df.columns[0]

    result = {"moead": [], "nsga": [], "comolsd": []}

    algo_to_key = {"MOEAD": "moead", "NSGAII": "nsga", "COMOLSD": "comolsd"}

    algo_values = df[ALGO_COLUMNS]
    # Linha a linha determina os algoritmos com menor valor
    for idx, row in algo_values.iterrows():
        values = row.to_dict()
        min_val = min(values.values())
        winners = [algo for algo, val in values.items() if val == min_val]
        if tie_strategy == "first":
            winners = winners[:1]
        instance_id = str(df.at[idx, instance_col])
        for algo in winners:
            result[algo_to_key[algo]].append(instance_id)

    return result


def build_all_indices(tie_strategy: str = "all") -> Dict[str, Dict[str, List[str]]]:
    """Monta o dicionário completo equivalente à estrutura usada em models_and_merit_builder."""
    full = {}
    for indicator in INDICATORS:
        full[indicator] = build_indices_for_indicator(indicator, tie_strategy=tie_strategy)
    return full


def as_python_literal(indices_dict: Dict[str, Dict[str, List[str]]]) -> str:
    """Gera uma string formatada como bloco Python para colar diretamente no código."""
    lines = []
    lines.append("indices_dict = {")
    for indicator, groups in indices_dict.items():
        lines.append(f"    '{indicator}': {{")
        for alg_key in ["nsga", "moead", "comolsd"]:
            if alg_key in groups:
                lines.append(f"        '{alg_key}': {groups[alg_key]},")
        lines.append("    },")
    lines.append("}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Gera dicionário de índices onde cada algoritmo é melhor.")
    parser.add_argument("--tie-strategy", choices=["all", "first"], default="all", help="Estratégia para empates.")
    parser.add_argument("--json", dest="json_path", help="Exporta resultado em JSON para o caminho dado.")
    parser.add_argument("--print-python", action="store_true", help="Imprime literal Python do dicionário.")
    args = parser.parse_args()

    indices = build_all_indices(tie_strategy=args.tie_strategy)

    if args.json_path:
        os.makedirs(os.path.dirname(args.json_path) or '.', exist_ok=True)
        with open(args.json_path, 'w', encoding='utf-8') as f:
            json.dump(indices, f, ensure_ascii=False, indent=2)
        print(f"JSON salvo em {args.json_path}")

    if args.print_python or not args.json_path:
        print(as_python_literal(indices))


if __name__ == "__main__":
    main()
