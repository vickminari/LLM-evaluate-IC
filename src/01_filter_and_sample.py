#!/usr/bin/env python3
"""
01_filter_and_sample.py
------------------------
Filtra chunks de baixa qualidade (tabelas quebradas na conversão, artigos
revogados/vetados sem conteúdo, fragmentos residuais) e faz amostragem
estratificada por domínio, documento-fonte e nível de dificuldade, gerando
as "tarefas de geração" que serão consumidas pelo script 02 (chamada aos
modelos via litellm).

USO:
    python 01_filter_and_sample.py \
        --input /mnt/user-data/uploads/all_chunks.jsonl \
        --output-dir ./out

SAÍDAS (em --output-dir):
    chunks_annotated.jsonl   -> todos os chunks originais + quality_flag
    generation_tasks.jsonl   -> tarefas amostradas, prontas para geração
    sampling_report.json     -> relatório de cobertura e descartes

Nada aqui é gerado por LLM: geração e validação humana ficam no script 02
e no fluxo de revisão manual, respectivamente.
"""

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------------------------------
# CONFIGURAÇÃO
# --------------------------------------------------------------------------

# Metas de pares QA finais por domínio
QA_TARGETS = {
    "legislacao": 150,
    "saude": 120,
    "edu": 120,
    "seguranca": 150,
}

# Proporção de itens por nível de dificuldade (soma = 1.0)
DIFFICULTY_SPLIT = {
    "factual": 0.36,
    "conceitual": 0.34,
    "aplicado": 0.30,
}

# Fator de sobre-amostragem: quantas "tarefas de geração" produzir por item
# final desejado, para compensar perdas na validação humana (itens
# descartados por baixa qualidade, duplicados, etc.)
OVERSAMPLE_FACTOR = 1.4

# Nenhum documento-fonte pode fornecer mais que esta fração das tarefas de
# um domínio (ajustado dinamicamente para domínios com poucos documentos)
MAX_SHARE_PER_DOCUMENT = 0.25

# Limiar de similaridade de cosseno (TF-IDF) para agrupar chunks
# tematicamente relacionados nos itens de nível "aplicado".
#
# ATENÇÃO (limitação conhecida): calibramos isso empiricamente em
# all_chunks.jsonl e a similaridade entre artigos jurídicos ficou alta mesmo
# sem relação temática real (mediana ~0.52), porque textos legais
# compartilham muito vocabulário institucional comum ("República
# Federativa", "Estados", "Distrito Federal" etc.). TF-IDF é lexical, não
# semântico -- ele não garante que os chunks agrupados sustentem uma
# pergunta de síntese coerente. Por isso o limiar abaixo já foi elevado
# para reduzir pares genéricos, mas TODO grupo de nível "aplicado" DEVE ser
# revisado manualmente antes da geração (ou pelo menos antes da validação
# final), descartando/reagrupando combinações sem relação real. Uma
# melhoria futura seria trocar TF-IDF por embeddings semânticos (ex.: um
# modelo de embedding local via Ollama), que discriminam melhor sinônimos e
# temas do que sobreposição lexical.
APLICADO_SIM_THRESHOLD = 0.35

# Stopwords em português (lista compacta) para reduzir o peso de conectores
# genéricos na similaridade TF-IDF
PT_STOPWORDS = list("""a ao aos as até com como da das de dela dele dentro depois do dos e ela elas
ele eles em entre era essa essas esse esses esta estas este estes eu foi for foram há isso isto já
lhe lhes mais mas me mesmo meu meus minha minhas muito na nas nem no nos nossa nossas nosso nossos
num numa o os ou para pela pelas pelo pelos por qual quando que quem se seu seus sua suas são só
também te tem tendo ter teu teus tu tua tuas um uma umas uns""".split())

SEED = 42

# --------------------------------------------------------------------------
# REGRAS DE QUALIDADE (calibradas empiricamente em all_chunks.jsonl)
# --------------------------------------------------------------------------

REVOKED_PATTERN = re.compile(r"\b(vetado|revogad[ao]s?)\b", re.IGNORECASE)


def classify_quality(chunk: dict) -> str:
    """Retorna um rótulo de qualidade para o chunk.

    Regras, em ordem de precedência:
      1. revoked_or_vetoed -> artigo revogado/vetado, sem conteúdo substantivo
      2. too_short         -> fragmento residual (< 6 palavras)
      3. fragmented_table  -> texto de tabela quebrado linha a linha na
                              conversão (poucas palavras por linha, muitas
                              linhas) -- ex.: PCDTs com quadros/tabelas
      4. ok                -> chunk utilizável para geração de perguntas
    """
    content = chunk["content"]
    word_count = chunk["word_count"]

    if REVOKED_PATTERN.search(content) and word_count < 15:
        return "revoked_or_vetoed"

    if word_count < 6:
        return "too_short"

    non_blank_lines = [l for l in content.split("\n") if l.strip()]
    n_lines = len(non_blank_lines)
    words_per_line = word_count / n_lines if n_lines else word_count
    if n_lines >= 4 and words_per_line < 4:
        return "fragmented_table"

    TABLE_HEADER_PATTERN = re.compile(r"^\s*(Tabela|Quadro|Gráfico|Grafico|Figura)\s*\d+(\.\d+)?", re.IGNORECASE)
    if word_count < 40 and ("Tabela" in content or "Fonte:" in content or "Sumário" in content or "CID" in content or "TABELA" in content or TABLE_HEADER_PATTERN.search(content)):
        return "table_artifact_or_short"

    return "ok"


# --------------------------------------------------------------------------
# AMOSTRAGEM COM COTA POR DOCUMENTO-FONTE
# --------------------------------------------------------------------------

def document_quotas(pool: list, n_target: int, max_share: float) -> dict:
    """Define quantos chunks vêm de cada source_document, respeitando um teto
    proporcional (max_share) para não deixar um documento grande dominar.

    Algoritmo: aloca proporcional ao nº de chunks disponíveis por documento,
    corta no teto, redistribui o excedente para os demais documentos
    (proporcionalmente), repetindo até estabilizar ou esgotar chunks.
    """
    by_doc = defaultdict(list)
    for c in pool:
        by_doc[c["source_document"]].append(c)

    docs = list(by_doc.keys())
    available = {d: len(by_doc[d]) for d in docs}
    effective_max_share = max(max_share, 1.0 / len(docs)) if docs else max_share
    cap = max(1, int(n_target * effective_max_share))

    quotas = {d: 0 for d in docs}
    remaining = n_target
    active_docs = set(docs)

    while remaining > 0 and active_docs:
        total_available = sum(available[d] for d in active_docs)
        if total_available == 0:
            break
        progressed = False
        for d in list(active_docs):
            if remaining <= 0:
                break
            share = available[d] / total_available
            alloc = min(
                available[d],
                cap - quotas[d],
                max(1, round(share * remaining)),
            )
            alloc = max(0, alloc)
            if alloc > 0:
                quotas[d] += alloc
                available[d] -= alloc
                remaining -= alloc
                progressed = True
            if quotas[d] >= cap or available[d] == 0:
                active_docs.discard(d)
        if not progressed:
            break

    return quotas, by_doc


def sample_single_chunks(pool: list, n_target: int, rng: random.Random) -> list:
    quotas, by_doc = document_quotas(pool, n_target, MAX_SHARE_PER_DOCUMENT)
    sampled = []
    for doc, q in quotas.items():
        candidates = by_doc[doc][:]
        rng.shuffle(candidates)
        sampled.extend(candidates[:q])
    rng.shuffle(sampled)
    return sampled[:n_target]


def build_tfidf_groups(pool: list, n_target_groups: int, rng: random.Random) -> list:
    """Agrupa chunks tematicamente relacionados (via TF-IDF + similaridade de
    cosseno) para os itens de nível 'aplicado', priorizando pares/trios que
    cruzam documentos-fonte diferentes (síntese real), com fallback para
    artigos vizinhos do mesmo documento quando não há par entre documentos.
    """
    if len(pool) < 3:
        return []

    texts = [c["content"] for c in pool]
    vectorizer = TfidfVectorizer(
        max_features=4096, ngram_range=(1, 2), stop_words=PT_STOPWORDS, min_df=2
    )
    matrix = vectorizer.fit_transform(texts)
    sim = cosine_similarity(matrix)

    n = len(pool)
    order = list(range(n))
    rng.shuffle(order)

    used = set()
    groups = []

    for i in order:
        if len(groups) >= n_target_groups:
            break
        if i in used:
            continue
        candidates = [
            (j, sim[i, j])
            for j in range(n)
            if j != i and j not in used and sim[i, j] >= APLICADO_SIM_THRESHOLD
        ]
        candidates.sort(key=lambda x: -x[1])

        cross_doc = [j for j, s in candidates if pool[j]["source_document"] != pool[i]["source_document"]]
        same_doc = [j for j, s in candidates if pool[j]["source_document"] == pool[i]["source_document"]]

        partners = cross_doc[:2] if cross_doc else same_doc[:1]
        if not partners:
            continue

        group_idx = [i] + partners
        group_type = "cross_source" if cross_doc else "same_source"

        used.update(group_idx)
        groups.append((group_idx, group_type))

    return [(([pool[k] for k in idxs]), gtype) for idxs, gtype in groups]


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Caminho para all_chunks.jsonl")
    parser.add_argument("--output-dir", default="./01_sampling_out", help="Diretório de saída")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Carrega e classifica qualidade -------------------------------------------------
    chunks = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))

    for c in chunks:
        c["quality_flag"] = classify_quality(c)

    annotated_path = out_dir / "chunks_annotated.jsonl"
    with open(annotated_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    quality_counts = defaultdict(int)
    for c in chunks:
        quality_counts[c["quality_flag"]] += 1

    # 2) Amostragem por domínio ---------------------------------------------------------
    tasks = []
    report = {"quality_counts": dict(quality_counts), "domains": {}}

    for domain, qa_target in QA_TARGETS.items():
        pool_ok = [c for c in chunks if c["domain"] == domain and c["quality_flag"] == "ok"]
        n_tasks_total = max(1, round(qa_target * OVERSAMPLE_FACTOR))

        n_factual = round(n_tasks_total * DIFFICULTY_SPLIT["factual"])
        n_conceitual = round(n_tasks_total * DIFFICULTY_SPLIT["conceitual"])
        n_aplicado = n_tasks_total - n_factual - n_conceitual

        rng.shuffle(pool_ok)
        used_ids = set()

        # --- Factual: 1 chunk por tarefa ---
        factual_candidates = [c for c in pool_ok if c["chunk_id"] not in used_ids]
        factual_sample = sample_single_chunks(factual_candidates, n_factual, rng)
        for c in factual_sample:
            used_ids.add(c["chunk_id"])
            tasks.append({
                "task_id": f"{domain}_factual_{len(tasks):04d}",
                "domain": domain,
                "nivel_dificuldade": "factual",
                "group_type": "single",
                "chunks": [strip_chunk(c)],
            })

        # --- Conceitual: 1 chunk por tarefa (o prompt do nível conceitual já
        #     instrui o gerador a explicar/comparar, não é preciso 2º chunk) ---
        conceitual_candidates = [c for c in pool_ok if c["chunk_id"] not in used_ids]
        conceitual_sample = sample_single_chunks(conceitual_candidates, n_conceitual, rng)
        for c in conceitual_sample:
            used_ids.add(c["chunk_id"])
            tasks.append({
                "task_id": f"{domain}_conceitual_{len(tasks):04d}",
                "domain": domain,
                "nivel_dificuldade": "conceitual",
                "group_type": "single",
                "chunks": [strip_chunk(c)],
            })

        # --- Aplicado: grupos de 2-3 chunks relacionados (via TF-IDF) ---
        aplicado_candidates = [c for c in pool_ok if c["chunk_id"] not in used_ids]
        aplicado_groups = build_tfidf_groups(aplicado_candidates, n_aplicado, rng)
        for group_chunks, gtype in aplicado_groups:
            for c in group_chunks:
                used_ids.add(c["chunk_id"])
            tasks.append({
                "task_id": f"{domain}_aplicado_{len(tasks):04d}",
                "domain": domain,
                "nivel_dificuldade": "aplicado",
                "group_type": gtype,
                "chunks": [strip_chunk(c) for c in group_chunks],
            })

        docs_covered = len({c["source_document"] for t in tasks if t["domain"] == domain for c in t["chunks"]})
        docs_available = len({c["source_document"] for c in pool_ok})

        report["domains"][domain] = {
            "qa_target": qa_target,
            "tasks_generated": {
                "factual": len(factual_sample),
                "conceitual": len(conceitual_sample),
                "aplicado": len(aplicado_groups),
            },
            "pool_ok_chunks": len(pool_ok),
            "pool_total_chunks": sum(1 for c in chunks if c["domain"] == domain),
            "source_documents_covered": docs_covered,
            "source_documents_available": docs_available,
        }

    # 3) Grava saídas --------------------------------------------------------------------
    tasks_path = out_dir / "generation_tasks.jsonl"
    with open(tasks_path, "w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    report_path = out_dir / "sampling_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Chunks totais: {len(chunks)}")
    print(f"Qualidade: {dict(quality_counts)}")
    print(f"Tarefas de geração amostradas: {len(tasks)}")
    print(f"-> {annotated_path}")
    print(f"-> {tasks_path}")
    print(f"-> {report_path}")


def strip_chunk(c: dict) -> dict:
    """Mantém só os campos relevantes para a geração (prompt) e rastreabilidade."""
    return {
        "chunk_id": c["chunk_id"],
        "domain": c["domain"],
        "source_document": c["source_document"],
        "article_ref": c.get("article_ref"),
        "section_title": c.get("section_title"),
        "hierarchy": c.get("hierarchy"),
        "content": c["content"],
    }


if __name__ == "__main__":
    main()
