#!/usr/bin/env python3
"""
patch_backfill_chunk_texto.py
--------------------------------
Corrige um bug do 08_evaluate_llms.py: o `chunk_texto` (trecho-fonte
original) nunca foi incluído em predictions.jsonl, então o LLM-judge
(09_compute_metrics.py) avaliou todas as respostas SEM acesso independente
à fonte -- só contra a resposta_referencia. Isso é um fator provável por
trás do viés de verbosidade encontrado na auditoria (Qwen Base, com
respostas ~5x mais longas, sendo julgado como "correto" ~14x mais vezes
que o esperado em relação ao Qwen FT nos casos de discordância).

Não precisa regerar as respostas dos modelos (caro: Qwen Base sozinho leva
horas) -- só recupera o chunk_texto via join pelo `id` contra o dataset
completo (govbench_br.jsonl, que tem o campo em todo item) e regrava um
predictions.jsonl pronto para rejulgar com 10_compute_metrics.py.

USO:
    python patch_backfill_chunk_texto.py \
        --scored scored_items.jsonl \
        --full-dataset govbench_br.jsonl \
        --output predictions_regrounded.jsonl

    python .\src\patch_backfill_chunk_texto.py --scored .\08_eval_out\predictions.jsonl \
    --full-dataset .\07_splits_out\govbench_br_validado_test.jsonl \
    --output .\08_eval_out\predictions_regrounded.jsonl
"""

import argparse
import json
from pathlib import Path


def load_jsonl(path: Path) -> list:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scored", required=True, help="scored_items.jsonl (ou predictions.jsonl) sem chunk_texto")
    parser.add_argument("--full-dataset", required=True, help="govbench_br.jsonl (ou equivalente) com chunk_texto por id")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    scored = load_jsonl(Path(args.scored))
    full = {r["id"]: r for r in load_jsonl(Path(args.full_dataset))}

    n_ok, n_missing = 0, 0
    out_rows = []
    for r in scored:
        full_item = full.get(r["id"])
        # mantém só os campos que o script 09 gera (descarta métricas já
        # calculadas -- vamos rejulgar do zero, ROUGE/F1/BERTScore não
        # precisam mudar mas o judge sim, então regravamos como
        # predictions.jsonl "limpo" para reentrar no pipeline do 10)
        row = {
            "id": r["id"], "dominio": r["dominio"], "nivel_dificuldade": r["nivel_dificuldade"],
            "model": r["model"], "pergunta": r["pergunta"],
            "resposta_referencia": r["resposta_referencia"],
            "resposta_gerada": r["resposta_gerada"],
            "latency_s": r.get("latency_s"), "error": r.get("error"),
        }
        if full_item and full_item.get("chunk_texto"):
            row["chunk_texto"] = full_item["chunk_texto"]
            n_ok += 1
        else:
            row["chunk_texto"] = []
            n_missing += 1
        out_rows.append(row)

    with open(args.output, "w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Itens com chunk_texto recuperado: {n_ok}")
    print(f"Itens SEM chunk_texto (id não encontrado): {n_missing}")
    print(f"-> {args.output}")
    print("\nPróximo passo: rode 09_compute_metrics.py de novo apontando pra este "
          "arquivo, com --judges (o cálculo de ROUGE-L/F1/BERTScore não muda, "
          "mas rode de novo mesmo assim -- é rápido -- para o scored_items.jsonl "
          "final já sair com tudo junto e coerente).")


if __name__ == "__main__":
    main()
