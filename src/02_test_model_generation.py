#!/usr/bin/env python3
"""
02_test_model_generation.py
-----------------------------
Roda um piloto de geração de perguntas sintéticas sobre as tarefas
amostradas pelo script 01, chamando um ou mais modelos (locais via Ollama
ou externos via API) através do litellm -- a mesma interface serve para
comparar, por exemplo, "ollama/gemma4:31b" com "gemini/gemini-3.6-flash"
sem mudar o código, só o nome do modelo.

USO:
    # ver os prompts que seriam enviados, sem gastar nada (sempre rode isso primeiro)
    python 02_test_model_generation.py --tasks ./out/generation_tasks.jsonl \
        --models ollama/gemma4:31b --n 5 --dry-run

    # piloto de verdade, comparando dois modelos em 10 tarefas cada
    python 02_test_model_generation.py --tasks ./out/generation_tasks.jsonl \
        --models ollama/gemma4:31b,gemini/gemini-3.6-flash --n 10

CONFIGURAÇÃO DE CREDENCIAIS (variáveis de ambiente, litellm lê sozinho):
    - Modelos locais via Ollama: nenhuma chave necessária; o Ollama precisa
      estar rodando (`ollama serve`) e o modelo já baixado (`ollama pull gemma4:31b`).
      Se o Ollama não estiver em localhost:11434, exporte OLLAMA_API_BASE.
    - Gemini:        export GEMINI_API_KEY="sua_chave"
    - OpenAI:        export OPENAI_API_KEY="sua_chave"
    - Outros provedores suportados pelo litellm seguem o mesmo padrão
      (ver https://docs.litellm.ai/docs/providers).

SAÍDA: um .jsonl por modelo em --output-dir, no schema combinado com o
GovBench-BR (id, dominio, fonte, chunk_id, chunk_texto, nivel_dificuldade,
pergunta, resposta_referencia, gerado_por, validado_por), mais um resumo
impresso no terminal (taxa de sucesso, latência média) para comparação
manual antes de rodar em escala.
"""

import argparse
import json
from pathlib import Path

from src.govbench_common import (
    SYSTEM_PROMPT,
    build_prompt,
    extract_json,
    call_model,
)

from dotenv import load_dotenv
load_dotenv()  # Carrega as variáveis do arquivo .env automaticamente


# --------------------------------------------------------------------------
# CHAMADA A UM MODELO PARA UMA TAREFA
# --------------------------------------------------------------------------

def run_one(model: str, task: dict, max_retries: int = 2, timeout: int = 120) -> dict:
    r = call_model(model=model, task=task, max_retries=max_retries, timeout=timeout)
    # Adapta para o schema do piloto
    r["task_id"] = task["task_id"]
    r["model"] = model
    r["domain"] = task["domain"]
    r["nivel_dificuldade"] = task["nivel_dificuldade"]
    r["chunk_ids"] = [c["chunk_id"] for c in task["chunks"]]
    return r


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", required=True, help="generation_tasks.jsonl do script 01")
    parser.add_argument("--models", required=True, help="modelos litellm separados por vírgula, ex.: ollama/gemma4:31b,gemini/gemini-3.6-flash")
    parser.add_argument("--n", type=int, default=10, help="nº de tarefas a testar por modelo (amostra do topo do arquivo)")
    parser.add_argument("--output-dir", default="./02_pilot_out")
    parser.add_argument("--dry-run", action="store_true", help="só mostra os prompts, não chama nenhum modelo")
    args = parser.parse_args()

    tasks = []
    with open(args.tasks, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))

    # amostra balanceada pelos 3 níveis de dificuldade, não só os primeiros N
    by_level = {"factual": [], "conceitual": [], "aplicado": []}
    for t in tasks:
        by_level[t["nivel_dificuldade"]].append(t)
    per_level = max(1, args.n // 3)
    sample = (
        by_level["factual"][:per_level]
        + by_level["conceitual"][:per_level]
        + by_level["aplicado"][: args.n - 2 * per_level]
    )

    if args.dry_run:
        print(f"[DRY RUN] {len(sample)} tarefas seriam enviadas. Exemplo de prompt completo:\n")
        print("=" * 70)
        print("SYSTEM:\n", SYSTEM_PROMPT)
        print("-" * 70)
        print("USER:\n", build_prompt(sample[0]))
        print("=" * 70)
        print(f"\n(mais {len(sample) - 1} tarefas seguiriam o mesmo formato)")
        return

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for model in models:
        print(f"\n=== Modelo: {model} ({len(sample)} tarefas) ===")
        results = []
        for i, task in enumerate(sample, start=1):
            r = run_one(model, task)
            results.append(r)
            status = "OK" if r["parse_ok"] else f"FALHOU ({r['error'][:60] if r['error'] else 'parse'})"
            print(f"  [{i}/{len(sample)}] {task['task_id']} ({task['nivel_dificuldade']}) -> {status}")

        safe_name = model.replace("/", "_").replace(":", "-")
        out_path = out_dir / f"pilot_{safe_name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        n_ok = sum(1 for r in results if r["parse_ok"])
        latencies = [r["latency_s"] for r in results if r["latency_s"] is not None]
        avg_lat = round(sum(latencies) / len(latencies), 2) if latencies else None
        print(f"  -> Sucesso de parsing: {n_ok}/{len(results)}  |  Latência média: {avg_lat}s")
        print(f"  -> Salvo em {out_path}")

    print(
        "\nPróximo passo sugerido: abrir os .jsonl gerados lado a lado e "
        "avaliar manualmente qual modelo produziu perguntas/respostas de "
        "melhor qualidade por nível de dificuldade, antes de rodar em escala."
    )


if __name__ == "__main__":
    main()
