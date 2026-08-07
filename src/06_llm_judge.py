#!/usr/bin/env python3
"""
06_llm_judge.py
-----------------
Camada 2 da auditoria: pede a um ou mais modelos-juiz (independentes do
modelo que gerou os dados, e independentes dos modelos que serão avaliados
no experimento final) para avaliar cada par pergunta/resposta contra o(s)
trecho(s)-fonte.

IMPORTANTE -- escolha do(s) juiz(es): NÃO use aqui nenhum modelo que também
apareça como (a) gerador do dataset (ex.: Gemma 4 31B) ou (b) modelo avaliado
no experimento final (ex.: Qwen3.5, LLaMA 3.1) -- em ambos os casos o juiz teria
incentivo/viés para aprovar o que ele mesmo produziria ou será testado nele.

USO (dois juízes, com sinalização de discordância):
    python 06_llm_judge.py --input govbench_br_clean.jsonl \
        --judges ollama/mistral-small,ollama/phi4 \
        --output-dir ./judge_out

SAÍDAS:
    govbench_judged.jsonl   -> cada item + veredito de cada juiz (retomável)
    priority_review.jsonl   -> subconjunto que precisa de atenção humana:
                               qualquer "rejeitar", qualquer "revisar", ou
                               discordância entre os juízes
    judge_summary.json      -> distribuição de veredito por juiz, taxa de
                               concordância, cruzamento com flags do script 05
                               (se o campo _flags/_overlap estiver presente)
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from src.govbench_common import call_judge


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
    parser.add_argument("--input", required=True, help="dataset GovBench-BR (já limpo dos descartes automáticos)")
    parser.add_argument(
        "--judges",
        default="ollama/command-r7b,ollama/phi4:14b",
        help="1+ modelos litellm separados por vírgula (padrão: ollama/command-r7b,ollama/phi4:14b)"
    )
    parser.add_argument("--output-dir", default="./06_llm_judge_out")
    parser.add_argument("--dominio", default=None)
    parser.add_argument("--nivel", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "govbench_judged.jsonl"

    items = load_jsonl(Path(args.input))
    if args.dominio:
        doms = set(d.strip() for d in args.dominio.split(","))
        items = [i for i in items if i["dominio"] in doms]
    if args.nivel:
        niveis = set(n.strip() for n in args.nivel.split(","))
        items = [i for i in items if i["nivel_dificuldade"] in niveis]

    judges = [j.strip() for j in args.judges.split(",") if j.strip()]

    # retomabilidade: pula itens já julgados por TODOS os juízes pedidos
    existing = load_jsonl(output_path)
    done_by_id = {row["id"]: row for row in existing}
    pending = [
        it for it in items
        if it["id"] not in done_by_id or not all(j in done_by_id[it["id"]].get("judge_verdicts", {}) for j in judges)
    ]
    if args.limit:
        pending = pending[: args.limit]

    print(f"Itens totais (após filtros): {len(items)}")
    print(f"A julgar nesta execução:     {len(pending)}")
    print(f"Juízes: {judges}\n")

    results = {row["id"]: row for row in existing}

    def save_current_results():
        with open(output_path, "w", encoding="utf-8") as out_f:
            for row in results.values():
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")

    for i, item in enumerate(pending, start=1):
        row = results.get(item["id"], {
            "id": item["id"], "dominio": item["dominio"],
            "nivel_dificuldade": item["nivel_dificuldade"],
            "judge_verdicts": {},
        })
        for judge_model in judges:
            if judge_model in row["judge_verdicts"]:
                continue
            r = call_judge(judge_model, item)
            row["judge_verdicts"][judge_model] = {
                "veredito": r["veredito"],
                "fundamentado_no_trecho": r["fundamentado_no_trecho"],
                "informacao_extra_nao_presente": r["informacao_extra_nao_presente"],
                "nivel_dificuldade_adequado": r["nivel_dificuldade_adequado"],
                "resposta_completa_precisa": r["resposta_completa_precisa"],
                "justificativa": r["justificativa"],
                "parse_ok": r["parse_ok"],
            }
        results[item["id"]] = row
        veredictos = {k: v["veredito"] for k, v in row["judge_verdicts"].items()}
        print(f"  [{i}/{len(pending)}] {item['id']} -> {veredictos}")
        
        # Salva incrementalmente a cada 5 itens (ou no último) para garantir integridade no disco
        if i % 5 == 0 or i == len(pending):
            save_current_results()

    # --- consolidação: concordância entre juízes + fila de prioridade -------------------
    priority = []
    all_verdicts_flat = []
    agree_count, total_multi = 0, 0

    for row in results.values():
        vs = {j: v["veredito"] for j, v in row["judge_verdicts"].items() if v.get("veredito")}
        all_verdicts_flat.extend(vs.values())
        needs_review = False
        if any(v in ("rejeitar", "revisar") for v in vs.values()):
            needs_review = True
        if len(judges) > 1 and len(set(vs.values())) > 1 and len(vs) == len(judges):
            needs_review = True
            total_multi += 1
        elif len(judges) > 1 and len(vs) == len(judges):
            total_multi += 1
            agree_count += 1
        if needs_review:
            priority.append(row)

    priority_path = out_dir / "priority_review.jsonl"
    with open(priority_path, "w", encoding="utf-8") as f:
        for row in priority:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "total_itens": len(results),
        "distribuicao_veredito_geral": dict(Counter(all_verdicts_flat)),
        "itens_para_revisao_prioritaria": len(priority),
        "taxa_concordancia_entre_juizes": round(agree_count / total_multi, 3) if total_multi else None,
    }
    with open(out_dir / "judge_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n-> {output_path}")
    print(f"-> {priority_path}  (revisar isto primeiro: {len(priority)} itens)")
    print(f"-> {out_dir / 'judge_summary.json'}")


if __name__ == "__main__":
    main()
