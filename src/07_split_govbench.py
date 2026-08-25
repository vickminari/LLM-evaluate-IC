#!/usr/bin/env python3
"""
07_split_govbench.py (corrigido)
-----------------------------------
Divide o GovBench-BR em treino/teste, estratificado por (domínio, nível de
dificuldade), SEM vazamento de dados por chunk compartilhado.

O QUE MUDOU em relação à v1: itens que compartilham qualquer chunk_id (ex.:
um item 'aplicado' que reusa um trecho também usado sozinho em um item
'factual') são agrupados em componentes conectados ANTES da divisão, e um
componente inteiro vai sempre para o mesmo lado (treino OU teste). Achado
real nos dados: 139 chunk_ids reaproveitados, afetando ~30% dos itens --
sem esse agrupamento, a v1 vazava esses itens entre os dois conjuntos.

Como um componente pode conter itens de mais de um nível de dificuldade
(mesmo domínio), a estratificação exata por estrato não é mais garantida
ao centavo -- o script faz uma alocação gulosa que minimiza o desvio em
relação à proporção alvo por estrato, e imprime o resultado real no final
para conferência. Termina com uma verificação de integridade (nenhum
chunk_id em treino E teste ao mesmo tempo) que interrompe a execução se
falhar.

USO: idêntico à v1.
    python 07_split_govbench.py --input govbench_br_validado.jsonl \
        --output-dir 07_splits_out --train-ratio 0.8 --seed 42
"""

import argparse
import hashlib
import json
import random
from collections import defaultdict, Counter
from pathlib import Path


def union_find_groups(items: list) -> list:
    """Agrupa itens em componentes conectados via chunk_id compartilhado."""
    parent = {it["id"]: it["id"] for it in items}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    chunk_to_first_item = {}
    for it in items:
        for cid in it["chunk_ids"]:
            if cid in chunk_to_first_item:
                union(it["id"], chunk_to_first_item[cid])
            else:
                chunk_to_first_item[cid] = it["id"]

    groups = defaultdict(list)
    for it in items:
        groups[find(it["id"])].append(it)
    return list(groups.values())


def group_sort_key(group: list) -> str:
    """Chave de ordenação determinística (hash), mais estável a pequenas
    mudanças no dataset do que um shuffle() dependente do tamanho da lista."""
    ids = sorted(it["id"] for it in group)
    return hashlib.md5(",".join(ids).encode()).hexdigest()


def split_govbench(input_filepath: str, train_ratio: float = 0.8, seed: int = 42, output_dir: str = "out/07_splits_out"):
    input_path = Path(input_filepath)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]
    print(f"Total de itens carregados: {len(items)}")

    groups = union_find_groups(items)
    multi_item_groups = [g for g in groups if len(g) > 1]
    n_affected = sum(len(g) for g in multi_item_groups)
    print(f"Componentes conectados (via chunk_id compartilhado): {len(groups)}")
    print(f"  -> {len(multi_item_groups)} componentes com >1 item, cobrindo {n_affected} itens\n")

    rng = random.Random(seed)
    groups_sorted = sorted(groups, key=group_sort_key)
    rng.shuffle(groups_sorted)  # desempata a ordem do hash de forma determinística pelo seed

    strata_keys = sorted(set((it["dominio"], it["nivel_dificuldade"]) for it in items))
    N_stratum = Counter((it["dominio"], it["nivel_dificuldade"]) for it in items)
    target_train = {s: round(N_stratum[s] * train_ratio) for s in strata_keys}

    assigned_train = Counter()
    assigned_total = Counter()

    train_items, test_items = [], []

    for group in groups_sorted:
        contrib = Counter((it["dominio"], it["nivel_dificuldade"]) for it in group)

        def cost(side):
            c = 0.0
            for s, n in contrib.items():
                new_train = assigned_train[s] + (n if side == "train" else 0)
                new_total = assigned_total[s] + n
                frac = new_train / new_total if new_total else 0.0
                c += n * (frac - train_ratio) ** 2
            return c

        side = "train" if cost("train") <= cost("test") else "test"
        for s, n in contrib.items():
            assigned_total[s] += n
            if side == "train":
                assigned_train[s] += n
        (train_items if side == "train" else test_items).extend(group)

    print("--- Distribuição por Estrato (Domínio x Dificuldade) ---")
    print(f"{'Domínio':<15} | {'Dificuldade':<12} | {'Total':<6} | {'Treino':<6} | {'Teste':<6} | {'% Treino':<8}")
    print("-" * 70)
    for s in strata_keys:
        dom, dif = s
        n_tr = assigned_train[s]
        n_tot = N_stratum[s]
        n_te = n_tot - n_tr
        pct = 100 * n_tr / n_tot if n_tot else 0
        print(f"{dom:<15} | {dif:<12} | {n_tot:<6} | {n_tr:<6} | {n_te:<6} | {pct:<8.1f}")
    print("-" * 70)
    print(f"TOTAL TREINO: {len(train_items)} ({len(train_items)/len(items)*100:.1f}%)")
    print(f"TOTAL TESTE:  {len(test_items)} ({len(test_items)/len(items)*100:.1f}%)")

    # --- verificação de integridade: nenhum chunk_id pode estar nos dois lados ------------
    train_chunks = {cid for it in train_items for cid in it["chunk_ids"]}
    test_chunks = {cid for it in test_items for cid in it["chunk_ids"]}
    overlap = train_chunks & test_chunks
    if overlap:
        raise RuntimeError(
            f"FALHA DE INTEGRIDADE: {len(overlap)} chunk_ids aparecem em treino E teste "
            f"simultaneamente (não deveria acontecer). Exemplos: {list(overlap)[:5]}"
        )
    print("\n[OK] Verificação de integridade: nenhum chunk_id compartilhado entre treino e teste.")

    train_path = out_dir / f"{input_path.stem}_train.jsonl"
    test_path = out_dir / f"{input_path.stem}_test.jsonl"
    with open(train_path, "w", encoding="utf-8") as f:
        for item in train_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    with open(test_path, "w", encoding="utf-8") as f:
        for item in test_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\nArquivos salvos com sucesso:")
    print(f"  - Treino: [train_file](file:///{train_path.resolve().as_posix()})")
    print(f"  - Teste:  [test_file](file:///{test_path.resolve().as_posix()})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Divide o dataset GovBench-BR em treino/teste, estratificado e sem vazamento por chunk compartilhado.")
    parser.add_argument("--input", default="out/05_cleaned_dataset_out/govbench_br.jsonl")
    parser.add_argument("--output-dir", default="out/07_splits_out")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    split_govbench(args.input, train_ratio=args.train_ratio, seed=args.seed, output_dir=args.output_dir)
