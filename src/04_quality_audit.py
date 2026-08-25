#!/usr/bin/env python3
"""
05_quality_audit.py
--------------------
Camada 1 (automática) + recomendação de amostra para Camada 3 (humana) da
auditoria de qualidade do GovBench-BR. Não substitui revisão humana nem
LLM-as-Judge (script separado) -- é o filtro rápido e gratuito que reduz
o volume que precisa de atenção mais cara.

Detecta:
  - duplicatas exatas de pergunta
  - quase-duplicatas (paráfrase da mesma pergunta, TF-IDF por domínio)
  - baixa fundamentação léxica resposta-vs-contexto (proxy de alucinação --
    NÃO é prova, é sinal para revisão manual)
  - chunks-fonte que parecem listas de referências bibliográficas (fonte
    conhecida de alucinação: não há conteúdo respondível, só citações)
  - itens 'aplicado' cuja resposta usa vocabulário de só 1 dos N trechos
    fornecidos (possível não-síntese real)
  - auto-referência vaga na pergunta (violação da diretriz de prompt)
  - inconsistência estrutural (nº de chunks incompatível com o nível)

USO:
    python 05_quality_audit.py --input govbench_br_raw_gemma4-31b.jsonl \
        --output-dir ./audit_out --confidence 0.90 --margin 0.10
"""

import argparse
import json
import re
import math
from collections import defaultdict, Counter
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --------------------------------------------------------------------------
CITE_PATTERN = re.compile(r"\d{4};\s*\d+\(\d+\):\s*\d+")
NUMBERED_REF_PATTERN = re.compile(r"^\s*\d{1,3}\.\s+[A-ZÀ-Ú]", re.MULTILINE)
VAGUE_PATTERN = re.compile(
    r"\b(segundo o trecho|de acordo com o trecho|conforme o trecho|"
    r"no trecho acima|conforme mencionado acima|de acordo com o texto acima|"
    r"com base no trecho fornecido)\b",
    re.IGNORECASE,
)
WORD_PATTERN = re.compile(r"[a-zà-ú]{4,}")

NEAR_DUP_THRESHOLD = 0.75
LOW_GROUNDING_THRESHOLD = 0.35
APLICADO_SINGLE_SOURCE_THRESHOLD = 0.15


def words(text: str) -> set:
    return set(WORD_PATTERN.findall(text.lower()))


def looks_like_reference_list(content: str) -> bool:
    return len(CITE_PATTERN.findall(content)) >= 2 or len(NUMBERED_REF_PATTERN.findall(content)) >= 3


def sample_size(N: int, confidence: float, margin: float) -> int:
    """Tamanho de amostra (Cochran, com correção para população finita)."""
    z_table = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    z = z_table.get(round(confidence, 2), 1.645)
    p = 0.5  # variância máxima (conservador)
    n0 = (z ** 2 * p * (1 - p)) / (margin ** 2)
    n = n0 / (1 + (n0 - 1) / N) if N > 0 else 0
    return max(1, min(N, math.ceil(n)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="out/04_quality_audit_out")
    parser.add_argument("--confidence", type=float, default=0.90, help="0.90, 0.95 ou 0.99")
    parser.add_argument("--margin", type=float, default=0.10, help="margem de erro, ex.: 0.10 = 10 p.p.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    for r in rows:
        r["_flags"] = []

    # 1) duplicatas exatas -------------------------------------------------------------
    seen = {}
    for r in rows:
        key = r["pergunta"].strip().lower()
        if key in seen:
            r["_flags"].append("exact_duplicate_question")
            rows_by_id = {rr["id"]: rr for rr in rows}
            rows_by_id[seen[key]]["_flags"].append("exact_duplicate_question")
        else:
            seen[key] = r["id"]

    # 2) quase-duplicatas por domínio (TF-IDF) ------------------------------------------
    by_domain = defaultdict(list)
    for r in rows:
        by_domain[r["dominio"]].append(r)

    near_dup_pairs = []
    for dom, items in by_domain.items():
        texts = [r["pergunta"] for r in items]
        vec = TfidfVectorizer(max_features=4096, ngram_range=(1, 2))
        m = vec.fit_transform(texts)
        sim = cosine_similarity(m)
        n = len(items)
        for i in range(n):
            for j in range(i + 1, n):
                if sim[i, j] > NEAR_DUP_THRESHOLD:
                    items[i]["_flags"].append("near_duplicate_question")
                    items[j]["_flags"].append("near_duplicate_question")
                    near_dup_pairs.append({
                        "domain": dom, "id_a": items[i]["id"], "id_b": items[j]["id"],
                        "similarity": round(float(sim[i, j]), 3),
                    })

    # 3) checagens por item --------------------------------------------------------------
    for r in rows:
        ctx = " ".join(r["chunk_texto"])
        resp_w = words(r["resposta_referencia"])
        ctx_w = words(ctx)
        overlap = (len(resp_w & ctx_w) / len(resp_w)) if resp_w else 0.0
        r["_overlap"] = round(overlap, 3)
        if overlap < LOW_GROUNDING_THRESHOLD:
            r["_flags"].append("low_lexical_grounding")

        if any(looks_like_reference_list(c) for c in r["chunk_texto"]):
            r["_flags"].append("reference_list_source")

        if VAGUE_PATTERN.search(r["pergunta"]):
            r["_flags"].append("vague_self_reference")

        n_chunks = len(r["chunk_ids"])
        if r["nivel_dificuldade"] == "aplicado" and n_chunks < 2:
            r["_flags"].append("chunk_count_mismatch")
        if r["nivel_dificuldade"] in ("factual", "conceitual") and n_chunks != 1:
            r["_flags"].append("chunk_count_mismatch")

        if r["nivel_dificuldade"] == "aplicado" and n_chunks >= 2:
            per_chunk_overlap = []
            for ct in r["chunk_texto"]:
                cw = words(ct)
                ov = (len(resp_w & cw) / len(resp_w)) if resp_w else 0
                per_chunk_overlap.append(ov)
            strong = [o for o in per_chunk_overlap if o > APLICADO_SINGLE_SOURCE_THRESHOLD]
            if len(strong) <= 1:
                r["_flags"].append("aplicado_single_source_reliance")

    # 4) grava saídas -----------------------------------------------------------------
    flagged = [r for r in rows if r["_flags"]]
    flag_counts = Counter(f for r in rows for f in r["_flags"])

    flagged_path = out_dir / "flagged_items.jsonl"
    with open(flagged_path, "w", encoding="utf-8") as f:
        for r in flagged:
            out_row = {k: v for k, v in r.items() if not k.startswith("chunk_texto")}
            f.write(json.dumps(out_row, ensure_ascii=False) + "\n")

    near_dup_path = out_dir / "near_duplicate_pairs.json"
    with open(near_dup_path, "w", encoding="utf-8") as f:
        json.dump(near_dup_pairs, f, ensure_ascii=False, indent=2)

    # 5) amostra estatística por estrato (domínio x nível), para revisão humana --------
    rng = np.random.default_rng(args.seed)
    review_sample_ids = set(r["id"] for r in flagged)  # 100% dos sinalizados
    sample_plan = {}
    for (dom, nivel), items in ((k, [r for r in rows if r["dominio"] == k[0] and r["nivel_dificuldade"] == k[1]])
                                  for k in sorted(set((r["dominio"], r["nivel_dificuldade"]) for r in rows))):
        N = len(items)
        n = sample_size(N, args.confidence, args.margin)
        unflagged = [r for r in items if not r["_flags"]]
        extra_needed = max(0, n - sum(1 for r in items if r["id"] in review_sample_ids))
        extra = rng.choice([r["id"] for r in unflagged], size=min(extra_needed, len(unflagged)), replace=False) if unflagged and extra_needed else []
        review_sample_ids.update(extra)
        sample_plan[f"{dom}__{nivel}"] = {"N": N, "amostra_recomendada": n, "sinalizados": sum(1 for r in items if r["_flags"])}

    sample_path = out_dir / "amostra_para_revisao_humana.jsonl"
    with open(sample_path, "w", encoding="utf-8") as f:
        for r in rows:
            if r["id"] in review_sample_ids:
                out_row = {k: v for k, v in r.items() if k != "_flags" or True}
                out_row["_motivo"] = "sinalizado" if r["_flags"] else "amostra_estatistica"
                f.write(json.dumps(out_row, ensure_ascii=False) + "\n")

    summary = {
        "total_itens": len(rows),
        "itens_sinalizados": len(flagged),
        "pct_sinalizados": round(100 * len(flagged) / len(rows), 1),
        "contagem_por_flag": dict(flag_counts),
        "pares_quase_duplicados": len(near_dup_pairs),
        "tamanho_total_amostra_revisao": len(review_sample_ids),
        "plano_amostral_por_estrato": sample_plan,
    }
    with open(out_dir / "audit_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n-> {flagged_path}")
    print(f"-> {near_dup_path}")
    print(f"-> {sample_path}  (revisar isto manualmente: {len(review_sample_ids)} itens)")
    print(f"-> {out_dir / 'audit_summary.json'}")


if __name__ == "__main__":
    main()
