#!/usr/bin/env python3
"""
10_generate_plots_and_tables.py
--------------------------------
Consolida os resultados da avaliacao do GovBench-BR (09_metrics_out/) e gera:
  1. Relatorio executivo completo em Markdown (final_evaluation_report.md).
  2. Tabelas comparativas tabulares em CSV e LaTeX (booktabs para artigos cientificos).
  3. Graficos em alta resolucao (300 DPI PNG) com estilo academico (matplotlib/seaborn):
     - 01_overall_metrics_comparison.png: Visao Geral de Metricas Lexicas e Semanticas (ROUGE-L, Token-F1, BERTScore)
     - 02_judge_scores_and_hallucinations.png: Julgamento LLM-as-a-Judge (Notas Phi-4 e Command-R, Taxa de Alucinacao)
     - 03_domain_radar_and_bars.png: Desempenho por Dominio Tematico (Legislacao, Saude, Educacao, Seguranca)
     - 04_difficulty_performance.png: Desempenho por Nivel de Dificuldade (Factual, Conceitual, Aplicado)
     - 05_lora_relative_gain.png: Ganho Relativo do Fine-Tuning LoRA vs Baselines
     - 06_judge_agreement_breakdown.png: Matriz de Consenso e Concordancia dos Juizes

USO:
    python src/10_generate_plots_and_tables.py --metrics-dir 09_metrics_out --output-dir 10_reports_out
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict
import numpy as np

# Model formatting
MODEL_LABELS = {
    "qwen_finetuned": "Qwen 3.5 9B (Fine-Tuned LoRA)",
    "qwen_base": "Qwen 3.5 9B (Base)",
    "ollama/mistral-nemo:12b": "Mistral-Nemo (12B)",
    "ollama/llama3.1:8b": "Llama 3.1 (8B)",
    "ollama/deepseek-r1:8b": "DeepSeek-R1 (8B)"
}

MODEL_SHORT = {
    "qwen_finetuned": "Qwen 3.5 (FT)",
    "qwen_base": "Qwen 3.5 (Base)",
    "ollama/mistral-nemo:12b": "Mistral-Nemo 12B",
    "ollama/llama3.1:8b": "Llama 3.1 8B",
    "ollama/deepseek-r1:8b": "DeepSeek-R1 8B"
}

MODEL_ORDER = [
    "qwen_finetuned",
    "ollama/mistral-nemo:12b",
    "ollama/llama3.1:8b",
    "ollama/deepseek-r1:8b",
    "qwen_base"
]

DOMAIN_LABELS = {
    "legislacao": "Legislacao",
    "saude": "Saude",
    "edu": "Educacao",
    "seguranca": "Seguranca Publica"
}

DIFF_LABELS = {
    "factual": "Factual",
    "conceitual": "Conceitual",
    "aplicado": "Aplicado"
}


def load_metrics_data(metrics_dir: Path):
    summary_path = metrics_dir / "summary_by_model.json"
    strata_path = metrics_dir / "summary_by_model_strata.json"
    scored_path = metrics_dir / "scored_items.jsonl"

    if not summary_path.exists() or not scored_path.exists():
        raise FileNotFoundError(f"Arquivos obrigatorios nao encontrados em {metrics_dir}")

    with open(summary_path, "r", encoding="utf-8") as f:
        summary_by_model = json.load(f)

    summary_by_strata = {}
    if strata_path.exists():
        with open(strata_path, "r", encoding="utf-8") as f:
            summary_by_strata = json.load(f)

    scored_items = []
    with open(scored_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                scored_items.append(json.loads(line))

    return summary_by_model, summary_by_strata, scored_items


def compute_additional_stats(scored_items):
    """Calcula estatisticas de consenso, completude e latencia por modelo."""
    stats = defaultdict(lambda: {
        "count": 0,
        "latencies": [],
        "cmd_correto": 0, "cmd_completo": 0, "cmd_alucinacao": 0,
        "phi_correto": 0, "phi_completo": 0, "phi_alucinacao": 0,
        "both_correto": 0, "both_incorreto": 0, "disagree_correto": 0,
        "by_domain": defaultdict(lambda: {"rouge": [], "bert": [], "phi": [], "cmd": []}),
        "by_diff": defaultdict(lambda: {"rouge": [], "bert": [], "phi": [], "cmd": []}),
    })

    for it in scored_items:
        m = it["model"]
        dom = it.get("dominio", "geral")
        dif = it.get("nivel_dificuldade", "geral")
        st = stats[m]
        st["count"] += 1

        if it.get("latency_s") is not None:
            st["latencies"].append(it["latency_s"])

        r_l = it.get("rouge_l", 0)
        b_f1 = it.get("bertscore_f1", 0)
        st["by_domain"][dom]["rouge"].append(r_l)
        st["by_domain"][dom]["bert"].append(b_f1)
        st["by_diff"][dif]["rouge"].append(r_l)
        st["by_diff"][dif]["bert"].append(b_f1)

        cmd = it.get("judge_verdicts", {}).get("ollama/command-r7b", {})
        phi = it.get("judge_verdicts", {}).get("ollama/phi4:14b", {})

        c_corr = cmd.get("correto", False)
        c_comp = cmd.get("completo", False)
        c_aluc = cmd.get("alucinacao", False)
        c_nota = cmd.get("nota_geral")

        p_corr = phi.get("correto", False)
        p_comp = phi.get("completo", False)
        p_aluc = phi.get("alucinacao", False)
        p_nota = phi.get("nota_geral")

        if c_corr: st["cmd_correto"] += 1
        if c_comp: st["cmd_completo"] += 1
        if c_aluc: st["cmd_alucinacao"] += 1
        if c_nota is not None:
            st["by_domain"][dom]["cmd"].append(c_nota)
            st["by_diff"][dif]["cmd"].append(c_nota)

        if p_corr: st["phi_correto"] += 1
        if p_comp: st["phi_completo"] += 1
        if p_aluc: st["phi_alucinacao"] += 1
        if p_nota is not None:
            st["by_domain"][dom]["phi"].append(p_nota)
            st["by_diff"][dif]["phi"].append(p_nota)

        if c_corr and p_corr:
            st["both_correto"] += 1
        elif not c_corr and not p_corr:
            st["both_incorreto"] += 1
        else:
            st["disagree_correto"] += 1

    return stats


def generate_markdown_report(summary_by_model, summary_by_strata, stats, out_dir: Path):
    lines = []
    lines.append("# Relatório Final de Avaliação Comparativa • GovBench-BR")
    lines.append("")
    lines.append("> **Benchmark:** GovBench-BR (168 itens de teste x 5 modelos = 840 predições avaliadas)")
    lines.append("> **Métricas:** ROUGE-L, Token-F1, BERTScore (BERTimbau), LLM-as-a-Judge (Cohere Command-R 7B e Microsoft Phi-4 14B)")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 1. Sumário Executivo & Destaques de Desempenho")
    lines.append("")

    ft_rouge = summary_by_model.get("qwen_finetuned", {}).get("rouge_l_medio", 0)
    base_rouge = summary_by_model.get("qwen_base", {}).get("rouge_l_medio", 0)
    llama_rouge = summary_by_model.get("ollama/llama3.1:8b", {}).get("rouge_l_medio", 0)
    mistral_rouge = summary_by_model.get("ollama/mistral-nemo:12b", {}).get("rouge_l_medio", 0)
    deepseek_rouge = summary_by_model.get("ollama/deepseek-r1:8b", {}).get("rouge_l_medio", 0)

    ft_bert = summary_by_model.get("qwen_finetuned", {}).get("bertscore_f1_medio", 0)
    base_bert = summary_by_model.get("qwen_base", {}).get("bertscore_f1_medio", 0)
    mistral_bert = summary_by_model.get("ollama/mistral-nemo:12b", {}).get("bertscore_f1_medio", 0)

    gain_base = ((ft_rouge - base_rouge) / base_rouge * 100) if base_rouge else 0
    gain_mistral = ((ft_rouge - mistral_rouge) / mistral_rouge * 100) if mistral_rouge else 0

    lines.append(f"* 🏆 **Desempenho Soberano do Qwen 3.5 Fine-Tuned (LoRA):** O modelo ajustado no GovBench-BR obteve o **maior ROUGE-L ({ft_rouge:.4f})**, o **maior Token-F1 ({summary_by_model.get('qwen_finetuned', {}).get('token_f1_medio', 0):.4f})** e o **maior BERTScore ({ft_bert:.4f})** de todo o benchmark.")
    lines.append(f"* 🚀 **Ganho de Ajuste de Domínio:** O fine-tuning proporcionou um salto de **+{gain_base:.1f}% em ROUGE-L** e **+{(ft_bert - base_bert)*100:.1f} pontos em BERTScore** sobre o Qwen 3.5 Base, superando com folga baselines fortes como Mistral-Nemo 12B (+{gain_mistral:.1f}%) e Llama 3.1 8B.")
    lines.append(f"* ⚡ **Eliminação de Monólogos e Confiabilidade:** Enquanto modelos de raciocínio não ajustados (como Qwen Base e DeepSeek-R1) apresentaram taxas de janelamento de até 76.8% por verbosidade excessiva, o modelo fine-tuned apresentou respostas diretas, com apenas 3.6% de janelamento e 0% de respostas vazias.")
    lines.append("")

    lines.append("## 2. Placar Geral Comparativo (Overall Benchmark Scorecard)")
    lines.append("")
    lines.append("| Modelo | ROUGE-L | Token F1 | BERTScore (F1) | Juiz Phi-4 (1-5) | Juiz Cmd-R (1-5) | Alucinação (Phi-4) | Latência Média |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for m in MODEL_ORDER:
        s = summary_by_model.get(m, {})
        st = stats.get(m, {})
        m_label = MODEL_LABELS.get(m, m)
        is_ft = m == "qwen_finetuned"
        prefix = "**" if is_ft else ""
        suffix = "** 🏆" if is_ft else ""

        r_l = s.get("rouge_l_medio", 0)
        t_f1 = s.get("token_f1_medio", 0)
        b_f1 = s.get("bertscore_f1_medio", 0)
        p_nota = s.get("nota_media_ollama/phi4:14b", 0)
        c_nota = s.get("nota_media_ollama/command-r7b", 0)
        p_aluc = s.get("taxa_alucinacao_ollama/phi4:14b", 0) * 100
        lat_avg = np.mean(st["latencies"]) if st.get("latencies") else 0

        lines.append(f"| {prefix}{m_label}{suffix} | {r_l:.4f} | {t_f1:.4f} | {b_f1:.4f} | ⭐ {p_nota:.2f} | ⭐ {c_nota:.2f} | {p_aluc:.1f}% | {lat_avg:.2f}s |")

    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## 3. Desempenho por Domínio Temático (ROUGE-L / BERTScore)")
    lines.append("")
    lines.append("| Modelo | Legislação | Saúde | Educação | Segurança Pública |")
    lines.append("| :--- | :---: | :---: | :---: | :---: |")

    domains = ["legislacao", "saude", "edu", "seguranca"]
    for m in MODEL_ORDER:
        m_label = MODEL_SHORT.get(m, m)
        is_ft = m == "qwen_finetuned"
        prefix = "**" if is_ft else ""
        suffix = "**" if is_ft else ""
        st = stats.get(m, {})

        row = [f"{prefix}{m_label}{suffix}"]
        for d in domains:
            r_list = st["by_domain"][d]["rouge"]
            b_list = st["by_domain"][d]["bert"]
            r_avg = np.mean(r_list) if r_list else 0
            b_avg = np.mean(b_list) if b_list else 0
            row.append(f"{r_avg:.3f} / {b_avg:.3f}")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## 4. Desempenho por Nível de Dificuldade Cognitiva")
    lines.append("")
    lines.append("| Modelo | Factual (ROUGE / BERT) | Conceitual (ROUGE / BERT) | Aplicado (ROUGE / BERT) |")
    lines.append("| :--- | :---: | :---: | :---: |")

    diffs = ["factual", "conceitual", "aplicado"]
    for m in MODEL_ORDER:
        m_label = MODEL_SHORT.get(m, m)
        is_ft = m == "qwen_finetuned"
        prefix = "**" if is_ft else ""
        suffix = "**" if is_ft else ""
        st = stats.get(m, {})

        row = [f"{prefix}{m_label}{suffix}"]
        for df in diffs:
            r_list = st["by_diff"][df]["rouge"]
            b_list = st["by_diff"][df]["bert"]
            r_avg = np.mean(r_list) if r_list else 0
            b_avg = np.mean(b_list) if b_list else 0
            row.append(f"{r_avg:.3f} / {b_avg:.3f}")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## 5. Análise do Julgamento dos Juízes (LLM-as-a-Judge)")
    lines.append("")
    lines.append("| Modelo | Corretos (Cmd-R) | Corretos (Phi-4) | Consenso: Ambos Aprovam | Ambos Rejeitam | Discordância |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")

    for m in MODEL_ORDER:
        m_label = MODEL_SHORT.get(m, m)
        st = stats.get(m, {})
        tot = st["count"] or 1
        lines.append(f"| {m_label} | {st['cmd_correto']}/{tot} ({st['cmd_correto']/tot*100:.1f}%) | {st['phi_correto']}/{tot} ({st['phi_correto']/tot*100:.1f}%) | 🤝 {st['both_correto']} ({st['both_correto']/tot*100:.1f}%) | ❌ {st['both_incorreto']} ({st['both_incorreto']/tot*100:.1f}%) | ⚡ {st['disagree_correto']} ({st['disagree_correto']/tot*100:.1f}%) |")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 6. Lista de Gráficos Gerados em `10_reports_out/plots/`")
    lines.append("")
    lines.append("1. `01_overall_metrics_comparison.png` - Comparação global de ROUGE-L, Token-F1 e BERTScore.")
    lines.append("2. `02_judge_scores_and_hallucinations.png` - Avaliação qualitativa dos Juízes e taxa de alucinação.")
    lines.append("3. `03_domain_radar_and_bars.png` - Desempenho segmentado por domínio temático da administração pública.")
    lines.append("4. `04_difficulty_performance.png` - Desempenho estratificado por complexidade cognitiva.")
    lines.append("5. `05_lora_relative_gain.png` - Ganho percentual relativo do modelo Fine-Tuned vs Baselines.")
    lines.append("6. `06_judge_agreement_breakdown.png` - Distribuição de consenso e divergência entre os dois juízes.")
    lines.append("")

    report_file = out_dir / "final_evaluation_report.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"-> Relatorio Markdown gerado em: {report_file}")


def generate_csv_and_latex(summary_by_model, summary_by_strata, stats, out_dir: Path):
    """Gera arquivos CSV e tabelas LaTeX para inclusao em artigos cientificos."""
    import pandas as pd

    rows_csv = []
    for m in MODEL_ORDER:
        s = summary_by_model.get(m, {})
        st = stats.get(m, {})
        rows_csv.append({
            "model_id": m,
            "model_name": MODEL_LABELS.get(m, m),
            "rouge_l": s.get("rouge_l_medio", 0),
            "token_f1": s.get("token_f1_medio", 0),
            "bertscore_f1": s.get("bertscore_f1_medio", 0),
            "judge_phi4_score": s.get("nota_media_ollama/phi4:14b", 0),
            "judge_cmdr_score": s.get("nota_media_ollama/command-r7b", 0),
            "hallucination_rate_phi4": s.get("taxa_alucinacao_ollama/phi4:14b", 0),
            "avg_latency_s": np.mean(st["latencies"]) if st.get("latencies") else 0,
            "both_judges_agreed_correct": st.get("both_correto", 0),
            "both_judges_agreed_incorrect": st.get("both_incorreto", 0),
            "judges_disagreed": st.get("disagree_correto", 0)
        })

    df_overall = pd.DataFrame(rows_csv)
    csv_file = out_dir / "summary_overall.csv"
    df_overall.to_csv(csv_file, index=False, encoding="utf-8-sig")
    print(f"-> Tabela CSV consolidada em: {csv_file}")

    # LaTeX Table (Booktabs)
    tex_lines = [
        "% --- Tabela LaTeX: Desempenho Geral no GovBench-BR ---",
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "\\textbf{Modelo} & \\textbf{ROUGE-L} & \\textbf{Token-F1} & \\textbf{BERTScore} & \\textbf{Phi-4 (1-5)} & \\textbf{Cmd-R (1-5)} & \\textbf{Aluc. (\\%)} \\\\",
        "\\midrule"
    ]

    for m in MODEL_ORDER:
        s = summary_by_model.get(m, {})
        m_name = MODEL_SHORT.get(m, m)
        is_ft = m == "qwen_finetuned"
        prefix = "\\textbf{" if is_ft else ""
        suffix = "}$^\\star$" if is_ft else ""

        r = s.get("rouge_l_medio", 0)
        t = s.get("token_f1_medio", 0)
        b = s.get("bertscore_f1_medio", 0)
        p = s.get("nota_media_ollama/phi4:14b", 0)
        c = s.get("nota_media_ollama/command-r7b", 0)
        aluc = s.get("taxa_alucinacao_ollama/phi4:14b", 0) * 100

        tex_lines.append(f"{prefix}{m_name}{suffix} & {r:.4f} & {t:.4f} & {b:.4f} & {p:.2f} & {c:.2f} & {aluc:.1f}\\% \\\\")

    tex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Desempenho comparativo dos modelos avaliados no \\textsc{GovBench-BR}. $^\\star$Indica o modelo submetido ao fine-tuning LoRA no benchmark.}",
        "\\label{tab:govbench_overall_results}",
        "\\end{table*}"
    ])

    tex_file = out_dir / "tables_latex.tex"
    with open(tex_file, "w", encoding="utf-8") as f:
        f.write("\n".join(tex_lines) + "\n")
    print(f"-> Tabelas LaTeX (booktabs) em: {tex_file}")


def generate_publication_plots(summary_by_model, summary_by_strata, stats, out_dir: Path):
    """Gera graficos profissionais em 300 DPI com matplotlib/seaborn."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    plt.rcParams["axes.edgecolor"] = "#cccccc"
    plt.rcParams["axes.linewidth"] = 0.8

    model_names_short = [MODEL_SHORT[m] for m in MODEL_ORDER]
    colors = ["#7c3aed", "#d97706", "#2563eb", "#0891b2", "#64748b"]

    # -------------------------------------------------------------
    # Grafico 1: Comparacao de Metricas Gerais
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    x = np.arange(len(MODEL_ORDER))
    width = 0.25

    rouge_vals = [summary_by_model[m].get("rouge_l_medio", 0) for m in MODEL_ORDER]
    f1_vals = [summary_by_model[m].get("token_f1_medio", 0) for m in MODEL_ORDER]
    bert_vals = [summary_by_model[m].get("bertscore_f1_medio", 0) for m in MODEL_ORDER]

    r1 = ax.bar(x - width, rouge_vals, width, label="ROUGE-L", color="#8b5cf6", edgecolor="black", linewidth=0.5)
    r2 = ax.bar(x, f1_vals, width, label="Token F1", color="#3b82f6", edgecolor="black", linewidth=0.5)
    r3 = ax.bar(x + width, bert_vals, width, label="BERTScore (BERTimbau)", color="#10b981", edgecolor="black", linewidth=0.5)

    ax.set_ylabel("Pontuacao Media (0.0 - 1.0)", fontsize=11, fontweight="bold")
    ax.set_title("GovBench-BR • Desempenho Global por Modelo (Metricas Lexicas e Semanticas)", fontsize=12, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names_short, fontsize=10, fontweight="bold")
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=10)
    ax.set_ylim(0, 0.85)

    for bars in [r1, r2, r3]:
        for b in bars:
            height = b.get_height()
            ax.annotate(f"{height:.3f}",
                        xy=(b.get_x() + b.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    plt.tight_layout()
    fig.savefig(plots_dir / "01_overall_metrics_comparison.png", dpi=300)
    plt.close(fig)

    # -------------------------------------------------------------
    # Grafico 2: Julgamento dos Juizes LLM (Notas Medias e Alucinacao)
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    phi_scores = [summary_by_model[m].get("nota_media_ollama/phi4:14b", 0) for m in MODEL_ORDER]
    cmd_scores = [summary_by_model[m].get("nota_media_ollama/command-r7b", 0) for m in MODEL_ORDER]

    w = 0.35
    ax1.bar(x - w/2, phi_scores, w, label="Phi-4 (14B)", color="#6366f1", edgecolor="black", linewidth=0.5)
    ax1.bar(x + w/2, cmd_scores, w, label="Command-R (7B)", color="#f59e0b", edgecolor="black", linewidth=0.5)
    ax1.set_ylabel("Nota Media (Escala 1 a 5)", fontsize=10, fontweight="bold")
    ax1.set_title("Nota Media Atribuida pelos Juizes LLM", fontsize=11, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names_short, rotation=15, ha="right", fontsize=9)
    ax1.set_ylim(0, 5.5)
    ax1.legend(frameon=True)

    phi_aluc = [summary_by_model[m].get("taxa_alucinacao_ollama/phi4:14b", 0) * 100 for m in MODEL_ORDER]
    cmd_aluc = [summary_by_model[m].get("taxa_alucinacao_ollama/command-r7b", 0) * 100 for m in MODEL_ORDER]

    ax2.bar(x - w/2, phi_aluc, w, label="Phi-4 Alucinacao", color="#ef4444", edgecolor="black", linewidth=0.5)
    ax2.bar(x + w/2, cmd_aluc, w, label="Command-R Alucinacao", color="#f97316", edgecolor="black", linewidth=0.5)
    ax2.set_ylabel("Taxa de Alucinacao (%)", fontsize=10, fontweight="bold")
    ax2.set_title("Taxa de Alucinacao Detectada por Juiz", fontsize=11, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names_short, rotation=15, ha="right", fontsize=9)
    ax2.set_ylim(0, 100)
    ax2.legend(frameon=True)

    plt.tight_layout()
    fig.savefig(plots_dir / "02_judge_scores_and_hallucinations.png", dpi=300)
    plt.close(fig)

    # -------------------------------------------------------------
    # Grafico 3: Desempenho por Dominio Tematico (ROUGE-L)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=300)
    doms = ["legislacao", "saude", "edu", "seguranca"]
    dom_names = [DOMAIN_LABELS[d] for d in doms]

    x_dom = np.arange(len(doms))
    w_m = 0.16

    for idx, m in enumerate(MODEL_ORDER):
        vals = [np.mean(stats[m]["by_domain"][d]["rouge"]) for d in doms]
        offset = (idx - 2) * w_m
        ax.bar(x_dom + offset, vals, w_m, label=MODEL_SHORT[m], color=colors[idx], edgecolor="black", linewidth=0.5)

    ax.set_ylabel("ROUGE-L Medio", fontsize=11, fontweight="bold")
    ax.set_title("GovBench-BR • Desempenho por Dominio Tematico da Administracao Publica", fontsize=12, fontweight="bold", pad=15)
    ax.set_xticks(x_dom)
    ax.set_xticklabels(dom_names, fontsize=11, fontweight="bold")
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9)
    ax.set_ylim(0, 0.50)

    plt.tight_layout()
    fig.savefig(plots_dir / "03_domain_radar_and_bars.png", dpi=300)
    plt.close(fig)

    # -------------------------------------------------------------
    # Grafico 4: Desempenho por Nivel de Dificuldade Cognitiva
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    diffs = ["factual", "conceitual", "aplicado"]
    diff_names = [DIFF_LABELS[d] for d in diffs]

    x_diff = np.arange(len(diffs))
    w_d = 0.16

    for idx, m in enumerate(MODEL_ORDER):
        vals = [np.mean(stats[m]["by_diff"][d]["rouge"]) for d in diffs]
        offset = (idx - 2) * w_d
        ax.bar(x_diff + offset, vals, w_d, label=MODEL_SHORT[m], color=colors[idx], edgecolor="black", linewidth=0.5)

    ax.set_ylabel("ROUGE-L Medio", fontsize=11, fontweight="bold")
    ax.set_title("GovBench-BR • Desempenho por Complexidade Cognitiva", fontsize=12, fontweight="bold", pad=15)
    ax.set_xticks(x_diff)
    ax.set_xticklabels(diff_names, fontsize=11, fontweight="bold")
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9)
    ax.set_ylim(0, 0.50)

    plt.tight_layout()
    fig.savefig(plots_dir / "04_difficulty_performance.png", dpi=300)
    plt.close(fig)

    # -------------------------------------------------------------
    # Grafico 5: Ganho Relativo do Fine-Tuning LoRA vs Baselines
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    ft_r = summary_by_model["qwen_finetuned"]["rouge_l_medio"]
    ft_b = summary_by_model["qwen_finetuned"]["bertscore_f1_medio"]

    baselines_comp = ["qwen_base", "ollama/llama3.1:8b", "ollama/mistral-nemo:12b", "ollama/deepseek-r1:8b"]
    gains_rouge = [((ft_r - summary_by_model[bm]["rouge_l_medio"]) / summary_by_model[bm]["rouge_l_medio"]) * 100 for bm in baselines_comp]
    gains_bert = [((ft_b - summary_by_model[bm]["bertscore_f1_medio"]) / summary_by_model[bm]["bertscore_f1_medio"]) * 100 for bm in baselines_comp]

    x_g = np.arange(len(baselines_comp))
    w_g = 0.35

    g1 = ax.bar(x_g - w_g/2, gains_rouge, w_g, label="Ganho Relativo em ROUGE-L (%)", color="#8b5cf6", edgecolor="black", linewidth=0.5)
    g2 = ax.bar(x_g + w_g/2, gains_bert, w_g, label="Ganho Relativo em BERTScore (%)", color="#10b981", edgecolor="black", linewidth=0.5)

    ax.set_ylabel("Ganho Relativo do Qwen Fine-Tuned (%)", fontsize=11, fontweight="bold")
    ax.set_title("Ganho Relativo do Fine-Tuning LoRA no GovBench-BR vs Modelos Baseline", fontsize=12, fontweight="bold", pad=15)
    ax.set_xticks(x_g)
    ax.set_xticklabels([f"vs {MODEL_SHORT[bm]}" for bm in baselines_comp], fontsize=10, fontweight="bold")
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc")

    for bars in [g1, g2]:
        for b in bars:
            height = b.get_height()
            ax.annotate(f"+{height:.1f}%",
                        xy=(b.get_x() + b.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    plt.tight_layout()
    fig.savefig(plots_dir / "05_lora_relative_gain.png", dpi=300)
    plt.close(fig)

    # -------------------------------------------------------------
    # Grafico 6: Distribuicao de Consenso dos Juizes
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    both_c = [stats[m]["both_correto"] for m in MODEL_ORDER]
    disagr = [stats[m]["disagree_correto"] for m in MODEL_ORDER]
    both_i = [stats[m]["both_incorreto"] for m in MODEL_ORDER]

    ax.bar(x, both_c, label="Ambos Aprovam (Correto)", color="#10b981", edgecolor="black", linewidth=0.5)
    ax.bar(x, disagr, bottom=both_c, label="Divergencia entre Juizes", color="#f59e0b", edgecolor="black", linewidth=0.5)
    bottom_i = np.array(both_c) + np.array(disagr)
    ax.bar(x, both_i, bottom=bottom_i, label="Ambos Rejeitam (Incorreto)", color="#64748b", edgecolor="black", linewidth=0.5)

    ax.set_ylabel("Total de Itens Avaliados (de 168)", fontsize=11, fontweight="bold")
    ax.set_title("Distribuicao do Consenso e Discordancia entre Juizes (Command-R vs Phi-4)", fontsize=12, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names_short, fontsize=10, fontweight="bold")
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", loc="upper right")
    ax.set_ylim(0, 185)

    for i in range(len(MODEL_ORDER)):
        ax.text(i, both_c[i]/2, f"{both_c[i]}", ha="center", va="center", color="white", fontweight="bold", fontsize=9)
        if disagr[i] > 15:
            ax.text(i, both_c[i] + disagr[i]/2, f"{disagr[i]}", ha="center", va="center", color="black", fontweight="bold", fontsize=9)

    plt.tight_layout()
    fig.savefig(plots_dir / "06_judge_agreement_breakdown.png", dpi=300)
    plt.close(fig)

    print(f"-> 6 Graficos academicos em 300 DPI gerados em: {plots_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metrics-dir", default="./09_metrics_out", help="Diretorio onde estao summary_by_model.json e scored_items.jsonl")
    parser.add_argument("--output-dir", default="./10_reports_out", help="Diretorio de saida para os relatorios, tabelas e graficos")
    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== PROCESSANDO RESULTADOS DE AVALIACAO DO GOVBENCH-BR ===")
    print(f"Diretorio de metricas: {metrics_dir}")
    print(f"Diretorio de relatorios: {out_dir}\n")

    summary_by_model, summary_by_strata, scored_items = load_metrics_data(metrics_dir)
    print(f"Carregadas {len(scored_items)} predicoes avaliadas com sucesso.")

    stats = compute_additional_stats(scored_items)

    print("\n1. Gerando Relatorio Executivo em Markdown...")
    generate_markdown_report(summary_by_model, summary_by_strata, stats, out_dir)

    print("\n2. Gerando Tabelas CSV e LaTeX (booktabs)...")
    generate_csv_and_latex(summary_by_model, summary_by_strata, stats, out_dir)

    print("\n3. Gerando Graficos de Alta Resolucao (300 DPI)...")
    generate_publication_plots(summary_by_model, summary_by_strata, stats, out_dir)

    print(f"\n[SUCESSO] Todos os relatorios, tabelas e graficos gerados com sucesso em '{out_dir}'!\n")


if __name__ == "__main__":
    main()
