#!/usr/bin/env python3
"""
10_compute_metrics.py
------------------------
Calcula ROUGE-L, F1 (nível de token), BERTScore e LLM-as-Judge sobre as
predições geradas pelo 09_evaluate_llms.py, agregando por modelo e por
estrato (domínio x nível de dificuldade).

JUÍZES RECOMENDADOS: Command-R7B + Phi-4 (os mesmos da curadoria do
benchmark, script 06). Se quiser um 3º juiz mais forte como desempate, 
prefira um modelo igualmente não-envolvido em nenhuma etapa anterior 
(ex.: Gemini via API), não um que já gerou ou validou o dataset.

USO:
    python 09_compute_metrics.py --predictions 08_eval_out/predictions.jsonl \
        --judges ollama/command-r7b,ollama/phi4 --output-dir 09_metrics_out

SAÍDAS:
    09_metrics_out/scored_items.jsonl   -> cada predição + todas as métricas
    09_metrics_out/summary_by_model.json         -> agregado por modelo
    09_metrics_out/summary_by_model_strata.json  -> agregado por modelo x domínio x nível
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from rouge_score import rouge_scorer
from govbench_common import call_eval_judge


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


# --------------------------------------------------------------------------
# MÉTRICAS LÉXICAS (sem dependência de modelo/GPU)
# --------------------------------------------------------------------------

_rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)


def compute_rouge_l(pred: str, ref: str) -> float:
    if not pred or not ref:
        return 0.0
    return _rouge.score(ref, pred)["rougeL"].fmeasure


def compute_token_f1(pred: str, ref: str) -> float:
    if not pred or not ref:
        return 0.0
    pred_tokens = pred.lower().split()
    ref_tokens = ref.lower().split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = {}
    for t in pred_tokens:
        common[t] = common.get(t, 0) + 1
    overlap = 0
    ref_counts = {}
    for t in ref_tokens:
        ref_counts[t] = ref_counts.get(t, 0) + 1
    for t, c in ref_counts.items():
        overlap += min(c, common.get(t, 0))
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


# --------------------------------------------------------------------------
# BERTSCORE (semântico, via BERTimbau -- carregado uma vez, em lote)
# --------------------------------------------------------------------------

def compute_bertscore_batch(rows: list, model_type: str = "neuralmind/bert-base-portuguese-cased") -> list:
    from bert_score import score as bert_score_fn

    preds = [r["resposta_gerada"] or "" for r in rows]
    refs = [r["resposta_referencia"] for r in rows]
    P, R, F1 = bert_score_fn(preds, refs, model_type=model_type, lang="pt", verbose=False)
    return F1.tolist()


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", required=True, help="predictions.jsonl do script 08")
    parser.add_argument("--judges", default="", help="modelos litellm separados por vírgula (vazio = pula LLM-judge)")
    parser.add_argument("--output-dir", default="09_metrics_out")
    parser.add_argument("--skip-bertscore", action="store_true", help="pula BERTScore (mais rápido p/ sanity check)")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scored_path = out_dir / "scored_items.jsonl"

    rows = load_jsonl(Path(args.predictions))
    rows = [r for r in rows if r.get("resposta_gerada")]  # ignora falhas de geração
    if args.limit:
        rows = rows[: args.limit]
    print(f"Itens a pontuar: {len(rows)}")

    # --- retomabilidade: pula (model,id) já pontuados com todos os juízes pedidos ---
    judges = [j.strip() for j in args.judges.split(",") if j.strip()]
    existing = load_jsonl(scored_path)
    done = {(r["model"], r["id"]): r for r in existing}

    print("1. Calculando ROUGE-L e F1 (léxicas)...")
    for r in rows:
        key = (r["model"], r["id"])
        if key in done and "rouge_l" in done[key]:
            r["rouge_l"] = done[key]["rouge_l"]
            r["token_f1"] = done[key]["token_f1"]
            continue
        r["rouge_l"] = round(compute_rouge_l(r["resposta_gerada"], r["resposta_referencia"]), 4)
        r["token_f1"] = round(compute_token_f1(r["resposta_gerada"], r["resposta_referencia"]), 4)

    if not args.skip_bertscore:
        print("2. Calculando BERTScore (BERTimbau, em lote -- pode demorar a 1ª vez p/ baixar o modelo)...")
        for r in rows:
            key = (r["model"], r["id"])
            if key in done and "bertscore_f1" in done[key]:
                r["bertscore_f1"] = done[key]["bertscore_f1"]
        to_score = [r for r in rows if "bertscore_f1" not in r]
        if to_score:
            f1s = compute_bertscore_batch(to_score)
            for r, f1 in zip(to_score, f1s):
                r["bertscore_f1"] = round(float(f1), 4)
    else:
        print("2. BERTScore pulado (--skip-bertscore)")
        for r in rows:
            r.setdefault("bertscore_f1", None)

    if judges:
        print(f"3. Rodando LLM-as-Judge ({judges})...")
        for i, r in enumerate(rows, start=1):
            r.setdefault("judge_verdicts", {})
            for j in judges:
                key = (r["model"], r["id"])
                if key in done and j in done[key].get("judge_verdicts", {}):
                    r["judge_verdicts"][j] = done[key]["judge_verdicts"][j]
                    continue
                jr = call_eval_judge(j, r)
                r["judge_verdicts"][j] = {
                    "correto": jr["correto"], "completo": jr["completo"],
                    "alucinacao": jr["alucinacao"], "nota_geral": jr["nota_geral"],
                    "justificativa": jr["justificativa"], "parse_ok": jr["parse_ok"],
                }
            if i % 20 == 0 or i == len(rows):
                print(f"   -> {i}/{len(rows)} itens julgados")
    else:
        print("3. LLM-as-Judge pulado (--judges vazio)")

    with open(scored_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # --- agregação -----------------------------------------------------------------
    def agg(group_rows: list) -> dict:
        n = len(group_rows)
        result = {
            "n_itens": n,
            "rouge_l_medio": round(sum(r.get("rouge_l", 0) for r in group_rows) / n, 4),
            "token_f1_medio": round(sum(r.get("token_f1", 0) for r in group_rows) / n, 4),
        }
        bs = [r["bertscore_f1"] for r in group_rows if r.get("bertscore_f1") is not None]
        if bs:
            result["bertscore_f1_medio"] = round(sum(bs) / len(bs), 4)
        for j in judges:
            notas = [r["judge_verdicts"][j]["nota_geral"] for r in group_rows
                     if r.get("judge_verdicts", {}).get(j, {}).get("nota_geral") is not None]
            alucinacoes = [r["judge_verdicts"][j]["alucinacao"] for r in group_rows
                           if r.get("judge_verdicts", {}).get(j, {}).get("alucinacao") is not None]
            if notas:
                result[f"nota_media_{j}"] = round(sum(notas) / len(notas), 2)
                result[f"taxa_alucinacao_{j}"] = round(sum(1 for a in alucinacoes if a) / len(alucinacoes), 3) if alucinacoes else None
        return result

    by_model = defaultdict(list)
    by_model_strata = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)
        by_model_strata[(r["model"], r["dominio"], r["nivel_dificuldade"])].append(r)

    summary_by_model = {m: agg(rs) for m, rs in by_model.items()}
    summary_by_model_strata = {f"{m}__{d}__{n}": agg(rs) for (m, d, n), rs in by_model_strata.items()}

    with open(out_dir / "summary_by_model.json", "w", encoding="utf-8") as f:
        json.dump(summary_by_model, f, ensure_ascii=False, indent=2)
    with open(out_dir / "summary_by_model_strata.json", "w", encoding="utf-8") as f:
        json.dump(summary_by_model_strata, f, ensure_ascii=False, indent=2)

    print("\n=== Resumo por modelo ===")
    print(json.dumps(summary_by_model, ensure_ascii=False, indent=2))
    print(f"\n-> {scored_path}")
    print(f"-> {out_dir / 'summary_by_model.json'}")
    print(f"-> {out_dir / 'summary_by_model_strata.json'}")


if __name__ == "__main__":
    main()
