#!/usr/bin/env python3
"""
05_clean_and_update_dataset.py
------------------------------
Aplica as regras de curadoria e sanitização no GovBench-BR:
  1. Remove itens marcados com 'reference_list_source' (fontes bibliográficas).
  2. Remove duplicatas exatas de pergunta.
  3. Remove quase-duplicatas de pergunta (manter id_a, remover id_b).
  4. Remove itens 'aplicado' com 'aplicado_single_source_reliance' (falta de síntese real).
  5. Expurga artefatos de tabela/quadro/gráfico e micro-chunks sem embasamento narrativo.
  6. Aplica descartes confirmados na curadoria humana (govbench_human_audit_decisions.json).
  7. Registra os itens expurgados em 04_quality_audit_out/removed_table_artifacts.jsonl.
  8. Salva o dataset limpo em 05_cleaned_dataset_out/govbench_br_raw_gemma4-31b_clean.jsonl.
"""

import json
import re
from collections import defaultdict
from pathlib import Path

TABLE_PATTERN = re.compile(r'\b(Tabela|Quadro|Gráfico|Grafico|Figura)\s*\d+(\.\d+)?\b', re.IGNORECASE)

def is_table_artifact(item: dict) -> tuple[bool, str]:
    q = item.get('pergunta', '')
    r = item.get('resposta_referencia', '')
    t = ' '.join(item.get('trechos_usados', []))
    c = ' '.join(item.get('chunk_texto', []))

    # Citação explícita a Tabela/Quadro/Gráfico/Figura
    match_q = TABLE_PATTERN.search(q)
    if match_q:
        return True, f"Citação explícita na pergunta: '{match_q.group(0)}'"

    match_r = TABLE_PATTERN.search(r)
    if match_r:
        return True, f"Citação explícita na resposta: '{match_r.group(0)}'"

    match_t = TABLE_PATTERN.search(t)
    if match_t:
        return True, f"Citação explícita nos trechos usados: '{match_t.group(0)}'"

    # Micro-chunk tabular sem narrativa (<35 palavras com indício de nota/tabela/sumário)
    words = len(c.split())
    if words < 35 and ('Tabela' in c or 'Fonte:' in c or 'Sumário' in c or 'CID' in c or 'TABELA' in c):
        return True, f"Micro-chunk tabular/nota ({words} palavras) sem narrativa"

    return False, ""

def main():
    root_dir = Path(__file__).resolve().parent.parent
    raw_path = root_dir / "03_generation_out" / "govbench_br_raw_gemma4-31b.jsonl"
    rep_path = root_dir / "03_generation_out" / "govbench_br_replacement_gemma4-31b.jsonl"
    clean_out_dir = root_dir / "05_cleaned_dataset_out"
    clean_out_dir.mkdir(parents=True, exist_ok=True)
    clean_path = clean_out_dir / "govbench_br_raw_gemma4-31b_clean.jsonl"
    
    flagged_path = root_dir / "04_quality_audit_out" / "flagged_items.jsonl"
    near_dups_path = root_dir / "04_quality_audit_out" / "near_duplicate_pairs.json"
    human_decisions_path = root_dir / "06_llm_judge_out" / "govbench_human_audit_decisions.json"
    removed_log_path = root_dir / "04_quality_audit_out" / "removed_table_artifacts.jsonl"
    
    items = []
    if raw_path.exists():
        with open(raw_path, 'r', encoding='utf-8') as f:
            items.extend([json.loads(l) for l in f if l.strip()])
            
    if rep_path.exists():
        with open(rep_path, 'r', encoding='utf-8') as f:
            items.extend([json.loads(l) for l in f if l.strip()])
            
    print(f"Total de itens lidos (Bruto + Reposição Pura): {len(items)}")
        
    flagged = []
    if flagged_path.exists():
        with open(flagged_path, 'r', encoding='utf-8') as f:
            flagged = [json.loads(l) for l in f if l.strip()]
            
    near_dups = []
    if near_dups_path.exists():
        with open(near_dups_path, 'r', encoding='utf-8') as f:
            near_dups = json.load(f)

    human_decisions = {}
    if human_decisions_path.exists():
        with open(human_decisions_path, 'r', encoding='utf-8') as f:
            human_decisions = json.load(f)
            
    removal_reasons = {}
    
    # 1. reference_list_source
    for i in flagged:
        if 'reference_list_source' in i.get('_flags', []):
            removal_reasons[i['id']] = "Fonte de lista bibliográfica (reference_list_source)"
            
    # 2. duplicatas exatas
    seen_exact = {}
    for i in items:
        p = i['pergunta'].strip().lower()
        if p in seen_exact:
            removal_reasons[i['id']] = f"Duplicata exata de pergunta de {seen_exact[p]}"
        else:
            seen_exact[p] = i['id']
            
    # 3. quase-duplicatas (remover id_b)
    for pair in near_dups:
        id_a, id_b = pair['id_a'], pair['id_b']
        if id_a not in removal_reasons:
            removal_reasons[id_b] = f"Quase-duplicata de {id_a} (simil: {pair.get('similarity', 0):.2f})"
        else:
            removal_reasons[id_a] = f"Quase-duplicata de {id_b} (simil: {pair.get('similarity', 0):.2f})"
            
    # 4. aplicado_single_source_reliance
    for i in flagged:
        if 'aplicado_single_source_reliance' in i.get('_flags', []):
            removal_reasons[i['id']] = "Nível aplicado sem síntese real (single_source_reliance)"

    # 5. Artefatos de Tabela / Micro-chunks tabulares
    for i in items:
        item_id = i['id']
        if item_id not in removal_reasons:
            is_tbl, reason = is_table_artifact(i)
            if is_tbl:
                removal_reasons[item_id] = f"Artefato de Tabela: {reason}"

    # 6. Descartes da Curadoria Humana
    for item_id, decision in human_decisions.items():
        if decision.get('action') == 'descartar' and item_id not in removal_reasons:
            notes = decision.get('notes', '').strip()
            notes_str = f" ({notes})" if notes else ""
            removal_reasons[item_id] = f"Descarte confirmado na curadoria humana{notes_str}"

    # Separa mantidos e descartados
    cleaned_items = []
    removed_items = []

    for item in items:
        item_id = item['id']
        if item_id in removal_reasons:
            removed_item = dict(item)
            removed_item['motivo_remocao'] = removal_reasons[item_id]
            removed_items.append(removed_item)
        else:
            cleaned_items.append(item)

    print("==========================================================")
    print("        RELATÓRIO DE HIGIENIZAÇÃO DO GOVBENCH-BR          ")
    print("==========================================================")
    print(f"Itens originais no dataset bruto:  {len(items)}")
    print(f"Itens expurgados/removidos:        {len(removed_items)}")
    print(f"Itens mantidos no dataset higienizado: {len(cleaned_items)}")
    
    # Distribuição por Domínio dos Removidos
    removed_by_domain = defaultdict(int)
    for r in removed_items:
        removed_by_domain[r['dominio']] += 1

    print("\n--- Descarte por Domínio ---")
    for dom, count in sorted(removed_by_domain.items()):
        print(f"  - {dom:<12}: {count} itens descartados")

    # Distribuição dos Mantidos
    kept_by_stratum = defaultdict(int)
    for k in cleaned_items:
        key = (k['dominio'], k['nivel_dificuldade'])
        kept_by_stratum[key] += 1

    print("\n--- Distribuição de Itens Mantidos por Estrato ---")
    print(f"{'Domínio':<15} | {'Dificuldade':<12} | {'Mantidos':<8}")
    print("-" * 42)
    for (dom, dif), count in sorted(kept_by_stratum.items()):
        print(f"{dom:<15} | {dif:<12} | {count:<8}")
    print("-" * 42)

    # Salva o arquivo de log dos expurgados
    with open(removed_log_path, 'w', encoding='utf-8') as f:
        for r in removed_items:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f"\nLog de itens expurgados salvo em: {removed_log_path}")

    # Salva o dataset higienizado final
    with open(clean_path, 'w', encoding='utf-8') as f:
        for item in cleaned_items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Dataset higienizado final salvo em: {clean_path}")

if __name__ == "__main__":
    main()
