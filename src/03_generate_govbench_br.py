#!/usr/bin/env python3
"""
03_generate_govbench_br.py
----------------------------
Geração OFICIAL do GovBench-BR em escala, a partir das tarefas amostradas
pelo script 01 e do modelo (ou modelos) escolhido no piloto (script 02).

Diferenças em relação ao piloto (02):
    - Roda todas as tarefas (ou um subconjunto filtrado por --dominio/--nivel),
      não só uma amostra de teste.
    - É RETOMÁVEL: se cair no meio, rodar de novo pula automaticamente as
      tarefas que já tiveram sucesso no arquivo de saída.
    - Escreve direto no schema final do GovBench-BR, pronto para a etapa de
      validação humana (validado_por continua null até a revisão manual --
      isso NUNCA é preenchido automaticamente por este script).
    - Falhas vão para um log separado (--failures), para reprocessar só
      elas depois, sem misturar com o dataset "bom".

FLUXO SUGERIDO (permite misturar modelos por domínio/nível, todos
escrevendo no mesmo arquivo de saída, de forma acumulativa):

    python 03_generate_govbench_br.py --tasks ./out/generation_tasks.jsonl \
        --model ollama/gemma4:31b --nivel factual,conceitual \
        --output ./out/govbench_br_raw.jsonl

    python 03_generate_govbench_br.py --tasks ./out/generation_tasks.jsonl \
        --model gemini/gemini-3.6-flash --nivel aplicado \
        --output ./out/govbench_br_raw.jsonl

    # se algo falhou, reprocessar só as falhas depois de investigar:
    python 03_generate_govbench_br.py --tasks ./out/generation_tasks.jsonl \
        --model ollama/gemma4:31b --output ./out/govbench_br_raw.jsonl

O arquivo de saída é o "GovBench-BR bruto": ainda precisa passar pela
validação humana obrigatória (seção 3.3.1) antes de virar o dataset final.
"""

import argparse
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()  # Carrega as variáveis do arquivo .env automaticamente

try:
    from govbench_common import call_model
except ModuleNotFoundError:
    from src.govbench_common import call_model


def load_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", required=True, help="generation_tasks.jsonl do script 01")
    parser.add_argument("--model", required=True, help="UM modelo litellm por execução, ex.: ollama/gemma4:31b")
    parser.add_argument("--output", required=True, help="govbench_br_raw.jsonl (acumulativo entre execuções)")
    parser.add_argument("--failures", default=None, help="log de falhas (default: <output>.failures.jsonl)")
    parser.add_argument("--dominio", default=None, help="filtro opcional, ex.: legislacao,saude")
    parser.add_argument("--nivel", default=None, help="filtro opcional, ex.: factual,conceitual,aplicado")
    parser.add_argument("--limit", type=int, default=None, help="processa no máximo N tarefas nesta execução")
    parser.add_argument("--sleep", type=float, default=0.0, help="pausa (s) entre chamadas, para respeitar rate limit de API")
    parser.add_argument("--workers", type=int, default=5, help="número de threads paralelas (padrão: 5)")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    failures_path = Path(args.failures) if args.failures else output_path.with_suffix(".failures.jsonl")

    all_tasks = load_jsonl(Path(args.tasks))
    already_done = set()
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    already_done.add(item["id"])

    pending = [t for t in all_tasks if t["task_id"] not in already_done]

    if args.dominio:
        domains = set(d.strip() for d in args.dominio.split(","))
        pending = [t for t in pending if t["domain"] in domains]
    if args.nivel:
        nivs = set(args.nivel.split(","))
        pending = [t for t in pending if t["nivel_dificuldade"] in nivs]
    if args.limit:
        pending = pending[: args.limit]

    print(f"Tarefas totais (após filtros): {len(all_tasks)}")
    print(f"Já concluídas (puladas):       {len(already_done)}")
    print(f"A processar nesta execução:    {len(pending)}")
    print(f"Modelo: {args.model}")
    print(f"Saída:  {output_path}")
    print(f"Falhas: {failures_path}")
    print(f"Workers paralelos: {args.workers}\n")

    n_ok, n_fail = 0, 0
    failures_this_run = []
    lock = threading.Lock()

    def process_task(task_tuple):
        idx, task = task_tuple
        r = call_model(args.model, task)
        if args.sleep:
            time.sleep(args.sleep)
        return idx, task, r

    with open(output_path, "a", encoding="utf-8") as out_f:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            task_tuples = [(i, t) for i, t in enumerate(pending, start=1)]
            for idx, task, r in executor.map(process_task, task_tuples):
                with lock:
                    if r["parse_ok"]:
                        row = {
                            "id": task["task_id"],
                            "dominio": task["domain"],
                            "nivel_dificuldade": task["nivel_dificuldade"],
                            "group_type": task["group_type"],
                            "fontes": [c["source_document"] for c in task["chunks"]],
                            "chunk_ids": [c["chunk_id"] for c in task["chunks"]],
                            "chunk_texto": [c["content"] for c in task["chunks"]],
                            "pergunta": r["pergunta"],
                            "resposta_referencia": r["resposta_referencia"],
                            "trechos_usados": r["trechos_usados"],
                            "gerado_por": args.model,
                            "validado_por": None,
                            "validado": False,
                        }
                        out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        out_f.flush()
                        n_ok += 1
                        status = "OK"
                    else:
                        failures_this_run.append({
                            "task_id": task["task_id"],
                            "domain": task["domain"],
                            "nivel_dificuldade": task["nivel_dificuldade"],
                            "model": args.model,
                            "error": r["error"],
                            "raw_response": r["raw_response"],
                        })
                        n_fail += 1
                        status = f"FALHOU ({(r['error'] or 'parse')[:60]})"

                    print(f"  [{n_ok + n_fail}/{len(pending)}] {task['task_id']} ({task['nivel_dificuldade']}) -> {status}")

    if failures_this_run:
        with open(failures_path, "w", encoding="utf-8") as f:
            for row in failures_this_run:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nConcluído: {n_ok} OK, {n_fail} falhas nesta execução.")
    if n_fail:
        print(f"Detalhes das falhas em: {failures_path}")
        print("Para reprocessar, rode este mesmo comando de novo (as que já "
              "deram certo são puladas automaticamente).")
    print(
        "\nLEMBRETE: este arquivo é o GovBench-BR BRUTO. Nenhum item aqui "
        "está validado (validado_por = null) -- a revisão humana obrigatória "
        "(seção 3.3.1) ainda precisa acontecer antes de usar em treino/avaliação."
    )


if __name__ == "__main__":
    main()
