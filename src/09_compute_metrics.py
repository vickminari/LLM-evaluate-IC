#!/usr/bin/env python3
"""
09_compute_metrics.py
------------------------
Calcula ROUGE-L, F1 (nível de token), BERTScore e LLM-as-Judge sobre as
predições geradas pelo 08_evaluate_llms.py, agregando por modelo e por
estrato (domínio x nível de dificuldade).

JUÍZES RECOMENDADOS: Command-R7B + Phi-4 (os mesmos da curadoria do benchmark, script 06)

USO:
    python 09_compute_metrics.py --predictions 08_eval_out/predictions.jsonl \
        --judges ollama/command-r7b,ollama/phi4:14b --output-dir 09_metrics_out

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

# --------------------------------------------------------------------------
# BERTSCORE (semântico, via BERTimbau) -- com janelamento por token para
# textos que excedem o limite de 512 tokens do BERT.
#
# Estratégia: NÃO trunca. Para o subconjunto de itens que excede o limite,
# divide o texto longo em janelas de até 510 tokens (via o próprio
# tokenizer do modelo, sem depender de separação por sentença -- evita
# quebrar em abreviações jurídicas como "Art." ou "Nº"), pontua cada janela
# contra o texto completo do outro lado, e usa o MÁXIMO entre as janelas
# (não a média): a referência é sempre curta e focada, então a média entre
# uma janela boa e várias janelas de "enchimento" penalizaria por
# verbosidade -- o que já é coberto pelos critérios `completo`/`alucinacao`
# do LLM-Judge. Aqui queremos só: "o conteúdo certo aparece em algum lugar?"
# --------------------------------------------------------------------------

MAX_BERT_TOKENS = 450  # margem de segurança contra subword re-tokenization e tokens especiais ([CLS]/[SEP])


def _needs_chunking(text: str, tokenizer) -> bool:
    return len(tokenizer.encode(text, add_special_tokens=False)) > MAX_BERT_TOKENS


def _chunk_text_by_tokens(text: str, tokenizer, max_len: int = MAX_BERT_TOKENS) -> list:
    """Divide o texto em janelas de até max_len tokens, usando o mecanismo
    nativo de overflow do tokenizer (limpo, sem depender de pontuação)."""
    enc = tokenizer(
        text, add_special_tokens=False, truncation=True, max_length=max_len,
        stride=0, return_overflowing_tokens=True,
    )
    return [tokenizer.decode(ids, skip_special_tokens=True) for ids in enc["input_ids"]]


def compute_bertscore_batch(rows: list, model_type: str = "neuralmind/bert-base-portuguese-cased", num_layers: int = 12) -> tuple:
    """Retorna (f1_scores, chunked_flags) -- um F1 e uma flag de "precisou
    janelar" por item de `rows`, na mesma ordem."""
    from bert_score import score as bert_score_fn
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_type)

    simple_idx, long_idx = [], []
    for i, r in enumerate(rows):
        pred, ref = r["resposta_gerada"] or "", r["resposta_referencia"]
        if _needs_chunking(pred, tokenizer) or _needs_chunking(ref, tokenizer):
            long_idx.append(i)
        else:
            simple_idx.append(i)

    f1_scores = [None] * len(rows)
    chunked_flags = [False] * len(rows)

    # --- caminho rápido: maioria dos itens, um único forward em lote ---
    if simple_idx:
        preds = [rows[i]["resposta_gerada"] or "" for i in simple_idx]
        refs = [rows[i]["resposta_referencia"] for i in simple_idx]
        _, _, F1 = bert_score_fn(preds, refs, model_type=model_type, num_layers=num_layers, verbose=False)
        for idx, f1 in zip(simple_idx, F1.tolist()):
            f1_scores[idx] = f1

    # --- itens longos: janela por token + máximo entre pares ---
    if long_idx:
        flat_preds, flat_refs, owner = [], [], []
        for i in long_idx:
            pred, ref = rows[i]["resposta_gerada"] or "", rows[i]["resposta_referencia"]
            pred_chunks = _chunk_text_by_tokens(pred, tokenizer) if _needs_chunking(pred, tokenizer) else [pred]
            ref_chunks = _chunk_text_by_tokens(ref, tokenizer) if _needs_chunking(ref, tokenizer) else [ref]
            for pc in pred_chunks:
                for rc in ref_chunks:
                    flat_preds.append(pc)
                    flat_refs.append(rc)
                    owner.append(i)
            chunked_flags[i] = True

        _, _, F1_long = bert_score_fn(flat_preds, flat_refs, model_type=model_type, num_layers=num_layers, verbose=False)
        best_by_owner = {}
        for i, f1 in zip(owner, F1_long.tolist()):
            best_by_owner[i] = max(f1, best_by_owner.get(i, -1))
        for i, f1 in best_by_owner.items():
            f1_scores[i] = f1

    return f1_scores, chunked_flags


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", required=True, help="predictions.jsonl do script 08")
    parser.add_argument("--judges", default="", help="modelos litellm separados por vírgula (vazio = pula LLM-judge)")
    parser.add_argument("--output-dir", default="09_metrics_out")
    parser.add_argument("--skip-bertscore", action="store_true", help="pula BERTScore (mais rápido p/ sanity check)")
    parser.add_argument("--overwrite", action="store_true", help="ignora scored_items.jsonl anterior e recalcula tudo do zero")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scored_path = out_dir / "scored_items.jsonl"

    all_rows = load_jsonl(Path(args.predictions))
    if args.limit:
        all_rows = all_rows[: args.limit]

    # taxa de resposta vazia por modelo, calculada ANTES de filtrar --
    # um modelo que trava/estoura o orçamento de tokens com frequência
    # (ex.: modelo 'thinking' sem tokens suficientes) não deve sumir
    # silenciosamente da média; isso é um resultado em si.
    empty_by_model = defaultdict(lambda: [0, 0])  # [vazios, total]
    for r in all_rows:
        empty_by_model[r["model"]][1] += 1
        if not r.get("resposta_gerada"):
            empty_by_model[r["model"]][0] += 1

    rows = [r for r in all_rows if r.get("resposta_gerada")]  # ignora falhas de geração
    print(f"Itens a pontuar: {len(rows)} (de {len(all_rows)} predições totais)")
    for m, (n_vazio, n_total) in empty_by_model.items():
        if n_vazio:
            print(f"   [AVISO] {m}: {n_vazio}/{n_total} respostas vazias "
                  f"({100*n_vazio/n_total:.1f}%) -- excluídas das métricas de conteúdo, "
                  f"mas contabilizadas em 'taxa_resposta_vazia' no resumo.")

    # --- retomabilidade: pula (model,id) já pontuados com todos os juízes pedidos ---
    judges = [j.strip() for j in args.judges.split(",") if j.strip()]
    if args.overwrite:
        done = {}
        print("   [OVERWRITE] Ignorando cache anterior e recalculando métricas e juízes do zero.")
    else:
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
                r["bertscore_chunked"] = done[key].get("bertscore_chunked", False)
        to_score = [r for r in rows if "bertscore_f1" not in r]
        if to_score:
            f1s, chunked_flags = compute_bertscore_batch(to_score)
            n_chunked = sum(chunked_flags)
            if n_chunked:
                print(f"   -> {n_chunked}/{len(to_score)} itens excederam 512 tokens e foram "
                      f"janelados (não truncados) para o BERTScore.")
            for r, f1, was_chunked in zip(to_score, f1s, chunked_flags):
                r["bertscore_f1"] = round(float(f1), 4)
                r["bertscore_chunked"] = was_chunked
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
            n_chunked = sum(1 for r in group_rows if r.get("bertscore_chunked"))
            if n_chunked:
                result["taxa_bertscore_janelado"] = round(n_chunked / len(bs), 3)
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
    for m, (n_vazio, n_total) in empty_by_model.items():
        if m in summary_by_model:
            summary_by_model[m]["taxa_resposta_vazia"] = round(n_vazio / n_total, 3)
            summary_by_model[m]["n_predicoes_totais"] = n_total
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
