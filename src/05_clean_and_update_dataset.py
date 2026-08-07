#!/usr/bin/env python3
"""
05_clean_and_update_dataset.py
------------------------------
Aplica as regras de curadoria e sanitização no GovBench-BR:
  1. Remove itens marcados com 'reference_list_source' (fontes bibliográficas).
  2. Remove duplicatas exatas de pergunta.
  3. Remove quase-duplicatas de pergunta (manter id_a, remover id_b).
  4. Remove itens 'aplicado' com 'aplicado_single_source_reliance' (falta de síntese real).
  5. Salva o dataset limpo e re-executa a divisão estratificada (80/20).
"""

import json
import os
from collections import defaultdict
from pathlib import Path

def main():
    root_dir = Path(__file__).resolve().parent.parent
    raw_path = root_dir / "03_generation_out" / "govbench_br_raw_gemma4-31b.jsonl"
    data_path = None
    
    flagged_path = root_dir / "04_quality_audit_out" / "flagged_items.jsonl"
    near_dups_path = root_dir / "04_quality_audit_out" / "near_duplicate_pairs.json"
    
    if not raw_path.exists():
        print(f"Erro: Arquivo {raw_path} não encontrado.")
        return
        
    with open(raw_path, 'r', encoding='utf-8') as f:
        items = [json.loads(l) for l in f if l.strip()]
        
    flagged = []
    if flagged_path.exists():
        with open(flagged_path, 'r', encoding='utf-8') as f:
            flagged = [json.loads(l) for l in f if l.strip()]
            
    near_dups = []
    if near_dups_path.exists():
        with open(near_dups_path, 'r', encoding='utf-8') as f:
            near_dups = json.load(f)
            
    to_remove = set()
    
    # 1. reference_list_source
    for i in flagged:
        if 'reference_list_source' in i.get('_flags', []):
            to_remove.add(i['id'])
            
    # 2. duplicatas exatas
    seen_exact = set()
    for i in items:
        p = i['pergunta'].strip().lower()
        if p in seen_exact:
            to_remove.add(i['id'])
        else:
            seen_exact.add(p)
            
    # 3. quase-duplicatas (remover id_b)
    for pair in near_dups:
        if pair['id_a'] not in to_remove:
            to_remove.add(pair['id_b'])
        else:
            to_remove.add(pair['id_a'])
            
    # 4. aplicado_single_source_reliance
    for i in flagged:
        if 'aplicado_single_source_reliance' in i.get('_flags', []):
            to_remove.add(i['id'])
            
    cleaned_items = [i for i in items if i['id'] not in to_remove]
    
    print(f"Itens originais: {len(items)}")
    print(f"Itens removidos:  {len(to_remove)}")
    print(f"Itens mantidos:   {len(cleaned_items)}")
    
    # Salva nos locais de saída
    for p in [raw_path, data_path]:
        if p and p.parent.exists():
            with open(p, 'w', encoding='utf-8') as f:
                for item in cleaned_items:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            print(f"Arquivo atualizado: {p}")

if __name__ == "__main__":
    main()
