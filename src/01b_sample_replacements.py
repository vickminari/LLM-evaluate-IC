#!/usr/bin/env python3
"""
01b_sample_replacements.py
--------------------------
Gera tarefas de substituição 100% narrativas de conteúdo puro (sem tabelas markdown,
sem eixos de gráficos, sem sumários com pontilhados '....', sem notas de rodapé/créditos,
sem CIDs e sem duplicatas de task_id) para repor os itens expurgados nos domínios de Segurança e Saúde.
"""

import json
import random
import re
from collections import defaultdict
from pathlib import Path

TABLE_PATTERN = re.compile(r'\b(Tabela|Quadro|Gráfico|Grafico|Figura)\s*\d+(\.\d+)?\b', re.IGNORECASE)

def is_pure_content_chunk(c: dict) -> bool:
    content = c.get("content", "")
    words = len(content.split())
    
    # 1. Tamanho mínimo razoável
    if words < 45:
        return False
        
    # 2. Elimina tabelas em formato Markdown pipe (|---| ou múltiplos |)
    if '|---|' in content or '| --- |' in content or content.count('|') > 3:
        return False
        
    # 3. Elimina menções a números de tabelas/quadros/gráficos
    if TABLE_PATTERN.search(content):
        return False
        
    # 4. Elimina artefatos de colunas, fontes e notas de rodapé
    if any(k in content for k in ['|Col1|', 'Col1', 'Col2', 'Fonte: SIM', 'Fonte: IBGE', 'Fonte: MS', 'CID 10']):
        return False
        
    # 5. Elimina Sumários / Índices com pontilhados (....) e metadados de capa/créditos
    if '....' in content or '...' in content or 'Sumário' in content or 'SUMÁRIO' in content:
        return False

    if content.count('@') >= 2 or 'Como referenciar:' in content or 'CDD 364' in content or 'ISBN' in content:
        return False

    # 6. Elimina eixos verticais de gráficos (listas de números/rótulos curtos no início do chunk)
    prefix = content[:350]
    prefix_lines = [l.strip() for l in prefix.split('\n') if l.strip()]
    short_lines = [l for l in prefix_lines if len(l) < 12]
    if len(prefix_lines) >= 6 and len(short_lines) / len(prefix_lines) > 0.6:
        return False

    return True

def main():
    repo = Path(__file__).resolve().parent.parent
    chunks_path = repo / "out" / "00_chunks_out" / "all_chunks.jsonl"
    clean_path = repo / "out" / "05_cleaned_dataset_out" / "govbench_br_raw_gemma4-31b_clean.jsonl"
    out_tasks_path = repo / "out" / "01_sampling_out" / "replacement_generation_tasks.jsonl"

    with open(clean_path, 'r', encoding='utf-8') as f:
        used_items = [json.loads(line) for line in f if line.strip()]

    used_chunk_ids = set()
    used_task_ids = set()
    for item in used_items:
        used_task_ids.add(item.get('task_id', item['id']))
        for cid in item.get('chunk_ids', []):
            used_chunk_ids.add(cid)

    with open(chunks_path, 'r', encoding='utf-8') as f:
        all_chunks = [json.loads(line) for line in f if line.strip()]

    # Alvos realistas baseados nos chunks puros disponíveis
    targets = {
        ('seguranca', 'factual'): 20,
        ('seguranca', 'conceitual'): 20,
        ('seguranca', 'aplicado'): 10,
        ('saude', 'factual'): 12,
        ('saude', 'conceitual'): 20,
        ('saude', 'aplicado'): 6,
    }

    pools = {'seguranca': [], 'saude': []}
    for c in all_chunks:
        d = c.get('domain')
        if d in pools and c['chunk_id'] not in used_chunk_ids:
            if is_pure_content_chunk(c):
                pools[d].append(c)

    random.seed(42)
    for d in pools:
        random.shuffle(pools[d])

    replacement_tasks = []
    stratum_counters = defaultdict(int)

    for (domain, level), target_n in targets.items():
        domain_chunks = pools[domain]
        collected = 0

        if level in ["factual", "conceitual"]:
            for c in domain_chunks:
                if collected >= target_n:
                    break
                stratum_counters[(domain, level)] += 1
                task_id = f"{domain}_{level}_pure_{stratum_counters[(domain, level)]:04d}"
                
                if task_id in used_task_ids:
                    continue
                used_task_ids.add(task_id)

                task = {
                    "task_id": task_id,
                    "domain": domain,
                    "nivel_dificuldade": level,
                    "group_type": "single",
                    "chunks": [c]
                }
                replacement_tasks.append(task)
                collected += 1
        elif level == "aplicado":
            i = 0
            while i < len(domain_chunks) - 1 and collected < target_n:
                c1 = domain_chunks[i]
                c2 = domain_chunks[i+1]
                i += 2

                stratum_counters[(domain, level)] += 1
                task_id = f"{domain}_{level}_pure_{stratum_counters[(domain, level)]:04d}"
                
                if task_id in used_task_ids:
                    continue
                used_task_ids.add(task_id)

                task = {
                    "task_id": task_id,
                    "domain": domain,
                    "nivel_dificuldade": level,
                    "group_type": "pair",
                    "chunks": [c1, c2]
                }
                replacement_tasks.append(task)
                collected += 1

        print(f"Estrato {domain} ({level}): {collected} / {target_n} tarefas de conteúdo puro criadas.")

    with open(out_tasks_path, 'w', encoding='utf-8') as f:
        for t in replacement_tasks:
            f.write(json.dumps(t, ensure_ascii=False) + '\n')

    print(f"\nTotal de tarefas de conteúdo puro salvas: {len(replacement_tasks)}")
    print(f"Salvo em: {out_tasks_path}")

if __name__ == "__main__":
    main()
