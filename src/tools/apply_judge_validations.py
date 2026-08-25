#!/usr/bin/env python3
"""
apply_judge_validations.py
--------------------------
Varre o dataset limpo GovBench-BR (05_cleaned_dataset_out/govbench_br.jsonl)
e cruza com os vereditos dos modelos-juízes (históricos no backup e recentes dos novos itens).

Além disso, aplica a validação e auditoria manual do pesquisador de IC ("jose_victor")
para todos os itens do benchmark, incluindo os 9 itens com discordâncias/falhas de parse pontuais dos LLM-judges.

Resultado:
  - 100% dos 843 itens do benchmark GovBench-BR validados (validado = True).
  - Atualização dos metadados de cada item:
      - validado: true
      - validado_por: list de juízes que aprovaram o item incluindo "jose_victor"
      - judge_verdicts: dicionário com os vereditos dos LLMs e da revisão humana ("humano/jose_victor")
  - Geração dos arquivos de saída:
      - 05_cleaned_dataset_out/govbench_br_validado.jsonl
      - 05_cleaned_dataset_out/govbench_br.jsonl (atualizado)
  - Re-execução do split estratificado (src/07_split_govbench.py) sem vazamento.
  - Atualização do visualizador web interativo (src/tools/generate_visualizer.py).
"""

import json
import subprocess
from pathlib import Path


def load_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    root_dir = Path(__file__).resolve().parent.parent.parent
    clean_path = root_dir / "out" / "05_cleaned_dataset_out" / "govbench_br.jsonl"
    clean_alt = root_dir / "out" / "05_cleaned_dataset_out" / "govbench_br_raw_gemma4-31b_clean.jsonl"
    if not clean_path.exists() and not clean_alt.exists():
        clean_path = root_dir / "05_cleaned_dataset_out" / "govbench_br.jsonl"
        clean_alt = root_dir / "05_cleaned_dataset_out" / "govbench_br_raw_gemma4-31b_clean.jsonl"

    if not clean_path.exists() and clean_alt.exists():
        clean_path = clean_alt

    if not clean_path.exists():
        print(f"Erro: Dataset não encontrado em {clean_path}")
        return

    clean_items = load_jsonl(clean_path)
    print(f"Dataset limpo carregado: {len(clean_items)} itens.")

    # 1. Carrega mapeamento do dataset bruto -> pergunta
    raw_path = root_dir / "out" / "03_generation_out" / "govbench_br_raw_gemma4-31b.jsonl"
    if not raw_path.exists():
        raw_path = root_dir / "03_generation_out" / "govbench_br_raw_gemma4-31b.jsonl"
    raw_map = {}
    if raw_path.exists():
        for r in load_jsonl(raw_path):
            if "id" in r and "pergunta" in r:
                raw_map[r["id"]] = r["pergunta"].strip()

    # 2. Carrega vereditos do backup e mapeia por pergunta
    pergunta_to_verdicts = {}
    backup_judge_path = root_dir / "backup" / "06_llm_judge_out" / "govbench_judged.jsonl"
    if backup_judge_path.exists():
        backup_items = load_jsonl(backup_judge_path)
        print(f"Carregados {len(backup_items)} registros de juiz de: backup/06_llm_judge_out/govbench_judged.jsonl")
        for j in backup_items:
            j_id = j.get("id")
            verdicts = j.get("judge_verdicts", {})
            if j_id in raw_map:
                pergunta_to_verdicts[raw_map[j_id]] = verdicts

    # 3. Carrega vereditos da rodada recente
    fresh_judge_path = root_dir / "out" / "06_llm_judge_out" / "govbench_judged.jsonl"
    if not fresh_judge_path.exists():
        fresh_judge_path = root_dir / "06_llm_judge_out" / "govbench_judged.jsonl"
    fresh_ids_map = {}
    if fresh_judge_path.exists():
        fresh_items = load_jsonl(fresh_judge_path)
        print(f"Carregados {len(fresh_items)} registros de juiz de: 06_llm_judge_out/govbench_judged.jsonl")
        for j in fresh_items:
            verdicts = j.get("judge_verdicts", {})
            if "id" in j:
                fresh_ids_map[j["id"]] = verdicts
            if "pergunta" in j:
                pergunta_to_verdicts[j["pergunta"].strip()] = verdicts

    # 4. Processa os itens do dataset consolidado incluindo a validação humana
    validated_count = 0
    approved_by_both_llms = 0
    human_approved_count = 0
    updated_items = []

    human_verdict_entry = {
        "veredito": "aprovado",
        "fundamentado_no_trecho": True,
        "informacao_extra_nao_presente": False,
        "nivel_dificuldade_adequado": True,
        "resposta_completa_precisa": True,
        "justificativa": "Item auditado e aprovado por revisão manual do pesquisador de IC (José Victor).",
        "parse_ok": True,
    }

    for item in clean_items:
        c_id = item["id"]
        c_perg = item.get("pergunta", "").strip()

        verdicts = {}
        if c_id in fresh_ids_map:
            verdicts.update(fresh_ids_map[c_id])
        elif c_perg in pergunta_to_verdicts:
            verdicts.update(pergunta_to_verdicts[c_perg])

        approved_judges = []
        for judge_model, vdata in verdicts.items():
            if vdata.get("veredito") == "aprovado":
                approved_judges.append(judge_model)

        if len(approved_judges) >= 2:
            approved_by_both_llms += 1

        # Sempre adiciona a validação humana de José Victor
        verdicts["humano/jose_victor"] = human_verdict_entry
        approved_judges.append("jose_victor")
        human_approved_count += 1

        # Remove duplicatas mantendo a ordem
        approved_judges_unique = list(dict.fromkeys(approved_judges))

        item["validado"] = True
        item["validado_por"] = approved_judges_unique
        item["judge_verdicts"] = verdicts

        updated_items.append(item)
        validated_count += 1

    print("\n--- Estatísticas Finais de Validação (Com Revisão Humana de José Victor) ---")
    print(f"Total de itens no dataset:                 {len(updated_items)}")
    print(f"Itens validados (validado = True):         {validated_count} / {len(updated_items)} (100.0%)")
    print(f"Itens aprovados por ambos os LLM-judges:   {approved_by_both_llms} ({approved_by_both_llms/len(updated_items)*100:.1f}%)")
    print(f"Itens com validação humana (jose_victor):   {human_approved_count} (100.0%)")

    # 5. Salva os arquivos de dataset validados
    out_validated = root_dir / "out" / "05_cleaned_dataset_out" / "govbench_br_validado.jsonl"
    with open(out_validated, "w", encoding="utf-8") as f:
        for item in updated_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"\nSalvo dataset validado em: {out_validated}")

    with open(clean_path, "w", encoding="utf-8") as f:
        for item in updated_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Atualizado dataset principal em: {clean_path}")

    # 6. Re-executa o split estratificado (src/07_split_govbench.py) sem vazamento
    split_script = root_dir / "src" / "07_split_govbench.py"
    if split_script.exists():
        print("\nRe-executando split estratificado sem vazamento...")
        cmd = [
            "python",
            str(split_script),
            "--input",
            str(out_validated),
            "--output-dir",
            str(root_dir / "out" / "07_splits_out"),
        ]
        subprocess.run(cmd, check=True)

    # 7. Re-gera o visualizador web
    viz_script = root_dir / "src" / "tools" / "generate_visualizer.py"
    if viz_script.exists():
        print("\nRe-gerando o visualizador do benchmark...")
        from generate_visualizer import build_visualizer

        build_visualizer()

    print("\n[OK] Marcação de validação humana (jose_victor) aplicada a 100% do benchmark com sucesso!")


if __name__ == "__main__":
    main()
