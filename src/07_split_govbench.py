import json
import random
from collections import defaultdict
from pathlib import Path

def split_govbench(
    input_filepath: str,
    train_ratio: float = 0.8,
    seed: int = 42,
    output_dir: str = "10_splits_out"
):
    random.seed(seed)
    input_path = Path(input_filepath)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(input_path, 'r', encoding='utf-8') as f:
        items = [json.loads(line) for line in f if line.strip()]
        
    print(f"Total de itens carregados: {len(items)}")
    
    # Agrupa por (dominio, nivel_dificuldade) para estratificação exata
    strata = defaultdict(list)
    for item in items:
        key = (item.get('dominio', 'desconhecido'), item.get('nivel_dificuldade', 'desconhecido'))
        strata[key].append(item)
        
    train_items = []
    test_items = []
    
    print("\n--- Distribuição por Estrato (Domínio x Dificuldade) ---")
    print(f"{'Domínio':<15} | {'Dificuldade':<12} | {'Total':<6} | {'Treino':<6} | {'Teste':<6}")
    print("-" * 55)
    
    for (dom, dif), stratum_items in sorted(strata.items()):
        # Embaralha os itens dentro do mesmo estrato
        random.shuffle(stratum_items)
        
        n_total = len(stratum_items)
        n_train = int(round(n_total * train_ratio))
        # Garante ao menos 1 exemplo no teste se houver mais de 1 item
        if n_train == n_total and n_total > 1:
            n_train = n_total - 1
        n_test = n_total - n_train
        
        stratum_train = stratum_items[:n_train]
        stratum_test = stratum_items[n_train:]
        
        train_items.extend(stratum_train)
        test_items.extend(stratum_test)
        
        print(f"{dom:<15} | {dif:<12} | {n_total:<6} | {len(stratum_train):<6} | {len(stratum_test):<6}")
        
    print("-" * 55)
    print(f"TOTAL TREINO: {len(train_items)} ({len(train_items)/len(items)*100:.1f}%)")
    print(f"TOTAL TESTE:  {len(test_items)} ({len(test_items)/len(items)*100:.1f}%)")
    
    # Salvar arquivos JSONL
    train_path = out_dir / f"{input_path.stem}_train.jsonl"
    test_path = out_dir / f"{input_path.stem}_test.jsonl"
    
    with open(train_path, 'w', encoding='utf-8') as f:
        for item in train_items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    with open(test_path, 'w', encoding='utf-8') as f:
        for item in test_items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    print(f"\nArquivos salvos com sucesso:")
    print(f"  - Treino: [train_file](file:///{train_path.resolve().as_posix()})")
    print(f"  - Teste:  [test_file](file:///{test_path.resolve().as_posix()})")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Divide o dataset GovBench-BR em treino (80%) e teste (20%) estratificado.")
    parser.add_argument("--input", default="03_generation_out/govbench_br_raw_gemma4-31b.jsonl", help="Dataset bruto ou limpo")
    parser.add_argument("--output-dir", default="07_splits_out", help="Diretório de saída para os splits")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Proporção de treino (padrão: 0.8)")
    parser.add_argument("--seed", type=int, default=42, help="Semente aleatória")
    args = parser.parse_args()

    split_govbench(args.input, train_ratio=args.train_ratio, seed=args.seed, output_dir=args.output_dir)
