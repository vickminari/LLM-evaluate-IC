#!/usr/bin/env python3
"""
standardize_ids.py
------------------
Padroniza e reordena sequencialmente os IDs de todos os 848 itens do 
GovBench-BR higienizado, removendo sufixos temporários como '_pure_' e 
garantindo numeração uniforme no formato:

  {dominio}_{nivel_dificuldade}_{index:04d}
  Exemplos: 
    seguranca_factual_0001 ... seguranca_factual_0069
    saude_aplicado_0001 ... saude_aplicado_0065

Ações executadas:
  1. Lê 05_cleaned_dataset_out/govbench_br_raw_gemma4-31b_clean.jsonl.
  2. Reordena e renumera os IDs sequencialmente por estrato (domínio x dificuldade).
  3. Salva o mapeamento antigo -> novo em 05_cleaned_dataset_out/id_renumbering_map.json.
  4. Sobrescreve o dataset limpo em 05_cleaned_dataset_out/govbench_br_raw_gemma4-31b_clean.jsonl.
  5. Atualiza o arquivo de vereditos do juiz em 06_llm_judge_out/govbench_judged.jsonl com os novos IDs.
  6. Re-executa o split estratificado (src/07_split_govbench.py).
  7. Re-gera o visualizador web em src/tools/benchmark_explorer.html.
"""

import json
import subprocess
from collections import defaultdict
from pathlib import Path

def standardize():
    root_dir = Path(__file__).resolve().parent.parent.parent
    clean_path = root_dir / "out" / "05_cleaned_dataset_out" / "govbench_br_raw_gemma4-31b_clean.jsonl"
    if not clean_path.exists():
        clean_path = root_dir / "05_cleaned_dataset_out" / "govbench_br_raw_gemma4-31b_clean.jsonl"
    map_path = root_dir / "out" / "05_cleaned_dataset_out" / "id_renumbering_map.json"
    judge_path = root_dir / "out" / "06_llm_judge_out" / "govbench_judged.jsonl"
    priority_path = root_dir / "out" / "06_llm_judge_out" / "priority_review.jsonl"
    
    if not clean_path.exists():
        print(f"Erro: Dataset higienizado não encontrado em {clean_path}")
        return

    with open(clean_path, 'r', encoding='utf-8') as f:
        items = [json.loads(line) for line in f if line.strip()]

    print(f"Total de itens para padronizar: {len(items)}")

    # Agrupa por (dominio, nivel_dificuldade)
    strata = defaultdict(list)
    for item in items:
        key = (item['dominio'], item['nivel_dificuldade'])
        strata[key].append(item)

    id_map = {}
    renumbered_items = []

    # Estratos em ordem alfabética para garantir reprodutibilidade
    for (dom, dif), stratum_items in sorted(strata.items()):
        # Mantém a ordem original do arquivo
        for idx, item in enumerate(stratum_items, start=1):
            old_id = item['id']
            new_id = f"{dom}_{dif}_{idx:04d}"
            item['id'] = new_id
            id_map[old_id] = new_id
            renumbered_items.append(item)

    print(f"Mapeamento de {len(id_map)} IDs construído com sucesso.")

    # 1. Salva o mapa de renumeração
    with open(map_path, 'w', encoding='utf-8') as f:
        json.dump(id_map, f, ensure_ascii=False, indent=2)

    # 2. Sobrescreve o dataset limpo com os novos IDs padronizados
    with open(clean_path, 'w', encoding='utf-8') as f:
        for item in renumbered_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Dataset limpo padronizado salvo em: {clean_path}")

    # 3. Atualiza os IDs no arquivo de vereditos do juiz (06_llm_judge_out/govbench_judged.jsonl) se existir
    if judge_path.exists():
        with open(judge_path, 'r', encoding='utf-8') as f:
            judged_items = [json.loads(line) for line in f if line.strip()]
        
        updated_judged = []
        for jitem in judged_items:
            old_id = jitem['id']
            if old_id in id_map:
                jitem['id'] = id_map[old_id]
                updated_judged.append(jitem)
        
        with open(judge_path, 'w', encoding='utf-8') as f:
            for jitem in updated_judged:
                f.write(json.dumps(jitem, ensure_ascii=False) + "\n")
        print(f"Arquivo de vereditos do juiz atualizado com novos IDs em: {judge_path}")

    # Atualiza priority_review.jsonl se existir
    if priority_path.exists():
        with open(priority_path, 'r', encoding='utf-8') as f:
            p_items = [json.loads(line) for line in f if line.strip()]
        updated_p = []
        for pitem in p_items:
            old_id = pitem['id']
            if old_id in id_map:
                pitem['id'] = id_map[old_id]
                updated_p.append(pitem)
        with open(priority_path, 'w', encoding='utf-8') as f:
            for pitem in updated_p:
                f.write(json.dumps(pitem, ensure_ascii=False) + "\n")

    # 4. Re-executa o split estratificado (07_split_govbench.py)
    split_script = root_dir / "src" / "07_split_govbench.py"
    splits_out_dir = root_dir / "out" / "07_splits_out"
    if split_script.exists():
        print("\nRe-executando split estratificado com os IDs padronizados...")
        cmd = [
            "python", str(split_script),
            "--input", str(clean_path),
            "--output-dir", str(splits_out_dir)
        ]
        subprocess.run(cmd, check=True)

    # 5. Re-gera o visualizador (src/tools/generate_visualizer.py)
    viz_script = root_dir / "src" / "tools" / "generate_visualizer.py"
    if viz_script.exists():
        print("\nRe-gerando o visualizador do benchmark...")
        from generate_visualizer import build_visualizer
        build_visualizer()

    print("\n[OK] Processo de padronização de IDs concluído com 100% de sucesso!")

if __name__ == "__main__":
    standardize()
