#!/usr/bin/env python3
"""
generate_eval_viewer.py
-----------------------
Gera uma interface web interativa estática (standalone HTML) para visualização, 
exploração comparativa e análise detalhada dos resultados das métricas e 
avaliações dos juízes (LLM-as-a-Judge) gerados pelo pipeline de avaliação.

Inclui:
- KPIs detalhados por modelo (ROUGE, F1, BERTScore, notas dos juízes, corretos, completos, alucinações).
- Estatísticas de concordância/discordância entre Command-R7B e Phi-4.
- Meta-avaliação humana dos juízes (voto 👍/👎 + anotação com persistência via localStorage e exportação JSON).
- Comparador Head-to-Head pergunta a pergunta.
"""

import json
from pathlib import Path

def build_eval_viewer():
    root_dir = Path(__file__).resolve().parent.parent.parent
    metrics_dir = root_dir / "out" / "09_metrics_out"
    if not metrics_dir.exists():
        metrics_dir = root_dir / "09_metrics_out"
    scored_path = metrics_dir / "scored_items.jsonl"
    summary_path = metrics_dir / "summary_by_model.json"
    strata_path = metrics_dir / "summary_by_model_strata.json"

    out_dir = Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_html = out_dir / "eval_metrics_viewer.html"

    if not scored_path.exists():
        print(f"Erro: Arquivo {scored_path} não encontrado!")
        return

    scored_items = []
    with open(scored_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                scored_items.append(json.loads(line))

    summary_by_model = {}
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            summary_by_model = json.load(f)

    summary_by_strata = {}
    if strata_path.exists():
        with open(strata_path, "r", encoding="utf-8") as f:
            summary_by_strata = json.load(f)

    print(f"Carregados {len(scored_items)} itens avaliados para compor o visualizador.")

    data_payload = {
        "items": scored_items,
        "summary": summary_by_model,
        "strata": summary_by_strata
    }

    json_data_str = json.dumps(data_payload, ensure_ascii=False)

    html_content = f"""<!DOCTYPE html>
<html lang="pt-BR" data-theme="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GovBench-BR | Dashboard de Avaliação e Meta-Avaliação de Juízes</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-main: #090d16;
            --bg-card: #111827;
            --bg-card-hover: #172033;
            --bg-input: #1f293d;
            --border-color: #28354f;
            --border-highlight: #3b82f6;
            --text-main: #f3f4f6;
            --text-muted: #9ca3af;
            --text-dim: #6b7280;

            --model-ft: #8b5cf6;
            --model-base: #64748b;
            --model-llama: #3b82f6;
            --model-mistral: #f59e0b;
            --model-deepseek: #06b6d4;

            --dom-leg: #ec4899;
            --dom-sau: #10b981;
            --dom-edu: #3b82f6;
            --dom-seg: #f59e0b;

            --badge-success: #10b981;
            --badge-warning: #f59e0b;
            --badge-danger: #ef4444;
            --badge-info: #06b6d4;
            --badge-purple: #8b5cf6;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
            background-color: var(--bg-main);
            color: var(--text-main);
            line-height: 1.5;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }}

        header {{
            background: linear-gradient(180deg, #131b2e 0%, rgba(19, 27, 46, 0.85) 100%);
            border-bottom: 1px solid var(--border-color);
            padding: 1rem 2rem;
            position: sticky;
            top: 0;
            z-index: 50;
            backdrop-filter: blur(12px);
        }}

        .header-container {{
            max-width: 1680px;
            margin: 0 auto;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 1rem;
        }}

        .brand {{
            display: flex;
            align-items: center;
            gap: 0.85rem;
        }}

        .brand-icon {{
            width: 42px;
            height: 42px;
            background: linear-gradient(135deg, #8b5cf6, #3b82f6);
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 800;
            font-size: 1.3rem;
            color: white;
            box-shadow: 0 4px 12px rgba(139, 92, 246, 0.35);
        }}

        .brand-text h1 {{
            font-size: 1.3rem;
            font-weight: 800;
            background: linear-gradient(90deg, #ffffff, #cbd5e1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}

        .brand-text p {{
            font-size: 0.8rem;
            color: var(--text-muted);
        }}

        .header-actions {{
            display: flex;
            align-items: center;
            gap: 0.75rem;
            flex-wrap: wrap;
        }}

        .audit-badge {{
            background: rgba(139, 92, 246, 0.15);
            border: 1px solid rgba(139, 92, 246, 0.4);
            color: #c4b5fd;
            padding: 0.4rem 0.85rem;
            border-radius: 8px;
            font-size: 0.8rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 0.4rem;
        }}

        .btn-export {{
            background: #8b5cf6;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.45rem 0.9rem;
            font-weight: 600;
            font-size: 0.82rem;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 0.4rem;
            transition: all 0.2s;
        }}

        .btn-export:hover {{
            background: #7c3aed;
            box-shadow: 0 4px 12px rgba(139, 92, 246, 0.4);
        }}

        .view-mode-tabs {{
            display: flex;
            background: var(--bg-input);
            padding: 4px;
            border-radius: 10px;
            border: 1px solid var(--border-color);
            gap: 4px;
        }}

        .tab-btn {{
            padding: 0.45rem 1rem;
            border-radius: 7px;
            border: none;
            background: transparent;
            color: var(--text-muted);
            font-weight: 600;
            font-size: 0.85rem;
            cursor: pointer;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            gap: 0.4rem;
        }}

        .tab-btn.active {{
            background: #3b82f6;
            color: white;
            box-shadow: 0 2px 8px rgba(59, 130, 246, 0.4);
        }}

        .tab-btn:hover:not(.active) {{
            color: var(--text-main);
            background: rgba(255, 255, 255, 0.05);
        }}

        main {{
            max-width: 1680px;
            margin: 0 auto;
            padding: 1.5rem 2rem 4rem 2rem;
            width: 100%;
            flex: 1;
        }}

        /* --- SUMMARY KPIS SECTION --- */
        .section-header-title {{
            font-size: 1.1rem;
            font-weight: 800;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            color: var(--text-main);
        }}

        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 1.25rem;
            margin-bottom: 2rem;
        }}

        .kpi-card {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 14px;
            padding: 1.25rem 1.4rem;
            position: relative;
            overflow: hidden;
            transition: all 0.25s ease;
            display: flex;
            flex-direction: column;
            gap: 0.85rem;
        }}

        .kpi-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
        }}

        .kpi-card.highlight {{
            border-color: #8b5cf6;
            background: linear-gradient(145deg, #171d33 0%, #1c183b 100%);
            box-shadow: 0 4px 20px rgba(139, 92, 246, 0.2);
        }}

        .kpi-card::before {{
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: var(--card-color, #64748b);
        }}

        .kpi-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .kpi-title {{
            font-size: 1rem;
            font-weight: 800;
            color: var(--text-main);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}

        .kpi-badge {{
            font-size: 0.7rem;
            padding: 0.15rem 0.5rem;
            border-radius: 6px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        .kpi-metrics-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 0.5rem;
        }}

        .kpi-metric-item {{
            background: rgba(0, 0, 0, 0.3);
            padding: 0.5rem 0.6rem;
            border-radius: 8px;
            border: 1px solid rgba(255, 255, 255, 0.05);
            text-align: center;
        }}

        .kpi-metric-label {{
            font-size: 0.68rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-weight: 600;
        }}

        .kpi-metric-val {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.05rem;
            font-weight: 700;
            color: var(--text-main);
            margin-top: 0.15rem;
        }}

        /* Detailed Judge Breakdown Inside KPI Card */
        .kpi-breakdown-section {{
            background: rgba(0, 0, 0, 0.2);
            border-radius: 8px;
            border: 1px solid var(--border-color);
            padding: 0.75rem;
            display: flex;
            flex-direction: column;
            gap: 0.6rem;
        }}

        .breakdown-row {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.76rem;
        }}

        .breakdown-title {{
            font-weight: 700;
            color: var(--text-muted);
            display: flex;
            align-items: center;
            gap: 0.3rem;
        }}

        .breakdown-chips {{
            display: flex;
            gap: 0.35rem;
        }}

        .b-chip {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem;
            font-weight: 700;
            padding: 0.15rem 0.4rem;
            border-radius: 4px;
        }}

        .b-ok {{ background: rgba(16, 185, 129, 0.15); color: #34d399; }}
        .b-comp {{ background: rgba(59, 130, 246, 0.15); color: #60a5fa; }}
        .b-aluc {{ background: rgba(239, 68, 68, 0.15); color: #f87171; }}

        .agreement-bar-container {{
            display: flex;
            flex-direction: column;
            gap: 0.25rem;
        }}

        .agreement-bar-label {{
            font-size: 0.72rem;
            color: var(--text-muted);
            display: flex;
            justify-content: space-between;
            font-weight: 600;
        }}

        .bar-track {{
            height: 8px;
            border-radius: 4px;
            background: rgba(255, 255, 255, 0.1);
            overflow: hidden;
            display: flex;
        }}

        .bar-seg {{
            height: 100%;
        }}

        .seg-both-ok {{ background: #10b981; }}
        .seg-both-fail {{ background: #64748b; }}
        .seg-disagree {{ background: #f59e0b; }}

        /* --- FILTER BAR --- */
        .filter-section {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 14px;
            padding: 1.25rem;
            margin-bottom: 1.75rem;
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }}

        .filter-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.85rem;
            align-items: center;
        }}

        .search-box {{
            flex: 1;
            min-width: 260px;
            position: relative;
        }}

        .search-input {{
            width: 100%;
            background: var(--bg-input);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 0.65rem 1rem 0.65rem 2.4rem;
            color: var(--text-main);
            font-size: 0.9rem;
            outline: none;
            transition: border-color 0.2s;
        }}

        .search-input:focus {{
            border-color: #8b5cf6;
            box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.25);
        }}

        .search-icon {{
            position: absolute;
            left: 0.85rem;
            top: 50%;
            transform: translateY(-50%);
            color: var(--text-dim);
            font-size: 0.95rem;
        }}

        .filter-group {{
            display: flex;
            align-items: center;
            gap: 0.4rem;
        }}

        .filter-label {{
            font-size: 0.78rem;
            font-weight: 600;
            color: var(--text-muted);
            white-space: nowrap;
        }}

        .select-input {{
            background: var(--bg-input);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 0.55rem 2rem 0.55rem 0.75rem;
            color: var(--text-main);
            font-size: 0.82rem;
            font-weight: 500;
            outline: none;
            cursor: pointer;
            appearance: none;
            background-image: url("data:image/svg+xml;charset=US-ASCII,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20width%3D%22292.4%22%20height%3D%22292.4%22%3E%3Cpath%20fill%3D%22%2394a3b8%22%20d%3D%22M287%2069.4a17.6%2017.6%200%200%200-13-5.4H18.4c-5%200-9.3%201.8-12.9%205.4A17.6%2017.6%200%200%200%200%2082.2c0%205%201.8%209.3%205.4%2012.9l128%20127.9c3.6%203.6%207.8%205.4%2012.8%205.4s9.2-1.8%2012.8-5.4L287%2095c3.5-3.5%205.4-7.8%205.4-12.8%200-5-1.9-9.2-5.5-12.8z%22%2F%3E%3C%2Fsvg%3E");
            background-repeat: no-repeat;
            background-position: right 0.65rem top 50%;
            background-size: 0.6rem auto;
        }}

        .filter-stats {{
            font-size: 0.82rem;
            color: var(--text-muted);
            margin-left: auto;
            font-weight: 600;
        }}

        /* --- ITEMS VIEW (CARDS) --- */
        .items-container {{
            display: flex;
            flex-direction: column;
            gap: 1.25rem;
        }}

        .item-card {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 14px;
            overflow: hidden;
            transition: border-color 0.2s, box-shadow 0.2s;
        }}

        .item-card:hover {{
            border-color: #3b82f6;
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.4);
        }}

        .item-header {{
            background: rgba(0, 0, 0, 0.25);
            border-bottom: 1px solid var(--border-color);
            padding: 0.85rem 1.25rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 0.75rem;
        }}

        .item-meta-left {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            flex-wrap: wrap;
        }}

        .item-id {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            font-weight: 700;
            color: var(--text-main);
            background: var(--bg-input);
            padding: 0.25rem 0.6rem;
            border-radius: 6px;
            border: 1px solid var(--border-color);
        }}

        .model-pill {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.8rem;
            font-weight: 700;
            padding: 0.25rem 0.65rem;
            border-radius: 6px;
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
        }}

        .model-qwen_finetuned {{ background: rgba(139, 92, 246, 0.2); color: #a78bfa; border: 1px solid rgba(139, 92, 246, 0.4); }}
        .model-qwen_base {{ background: rgba(100, 116, 139, 0.2); color: #94a3b8; border: 1px solid rgba(100, 116, 139, 0.4); }}
        .model-llama3_1_8b {{ background: rgba(59, 130, 246, 0.2); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.4); }}
        .model-mistral_nemo_12b {{ background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4); }}
        .model-deepseek_r1_8b {{ background: rgba(6, 182, 212, 0.2); color: #22d3ee; border: 1px solid rgba(6, 182, 212, 0.4); }}

        .dom-pill {{
            font-size: 0.75rem;
            font-weight: 700;
            padding: 0.2rem 0.55rem;
            border-radius: 6px;
            text-transform: capitalize;
        }}

        .dom-legislacao {{ background: rgba(236, 72, 153, 0.15); color: #f472b6; }}
        .dom-saude {{ background: rgba(16, 185, 129, 0.15); color: #34d399; }}
        .dom-edu {{ background: rgba(59, 130, 246, 0.15); color: #60a5fa; }}
        .dom-seguranca {{ background: rgba(245, 158, 11, 0.15); color: #fbbf24; }}

        .diff-pill {{
            font-size: 0.72rem;
            font-weight: 600;
            padding: 0.2rem 0.5rem;
            border-radius: 6px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            background: var(--bg-input);
            color: var(--text-muted);
        }}

        .item-metrics-bar {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            flex-wrap: wrap;
        }}

        .metric-badge {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.78rem;
            padding: 0.25rem 0.55rem;
            border-radius: 6px;
            background: var(--bg-input);
            border: 1px solid var(--border-color);
            display: flex;
            align-items: center;
            gap: 0.35rem;
        }}

        .metric-badge span.val {{
            font-weight: 700;
        }}

        .metric-high {{ color: #10b981; border-color: rgba(16, 185, 129, 0.4); }}
        .metric-mid {{ color: #f59e0b; border-color: rgba(245, 158, 11, 0.4); }}
        .metric-low {{ color: #ef4444; border-color: rgba(239, 68, 68, 0.4); }}

        .item-body {{
            padding: 1.25rem;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.25rem;
        }}

        @media (max-width: 1024px) {{
            .item-body {{
                grid-template-columns: 1fr;
            }}
        }}

        .content-box {{
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
        }}

        .content-label {{
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--text-muted);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}

        .box-text {{
            background: rgba(0, 0, 0, 0.25);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 0.85rem 1rem;
            font-size: 0.88rem;
            color: var(--text-main);
            white-space: pre-wrap;
            word-break: break-word;
            flex: 1;
        }}

        .box-text.question {{
            border-left: 3px solid #3b82f6;
            font-weight: 500;
        }}

        .box-text.reference {{
            border-left: 3px solid #10b981;
            background: rgba(16, 185, 129, 0.05);
        }}

        .box-text.generated {{
            border-left: 3px solid #8b5cf6;
            background: rgba(139, 92, 246, 0.05);
        }}

        /* --- JUDGES SECTION & HUMAN META-EVALUATION --- */
        .judges-section {{
            grid-column: 1 / -1;
            border-top: 1px solid var(--border-color);
            padding-top: 1.25rem;
            margin-top: 0.25rem;
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
        }}

        .judges-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(380px, 1fr));
            gap: 1.25rem;
        }}

        .judge-box {{
            background: var(--bg-input);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1rem 1.1rem;
            display: flex;
            flex-direction: column;
            gap: 0.65rem;
            position: relative;
        }}

        .judge-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .judge-name {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            font-weight: 700;
            color: var(--text-main);
        }}

        .judge-stars {{
            color: #f59e0b;
            font-size: 0.95rem;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 0.25rem;
        }}

        .judge-flags {{
            display: flex;
            gap: 0.4rem;
            flex-wrap: wrap;
        }}

        .flag-pill {{
            font-size: 0.72rem;
            font-weight: 700;
            padding: 0.15rem 0.45rem;
            border-radius: 4px;
        }}

        .flag-ok {{ background: rgba(16, 185, 129, 0.2); color: #34d399; }}
        .flag-bad {{ background: rgba(239, 68, 68, 0.2); color: #f87171; }}
        .flag-neutral {{ background: rgba(100, 116, 139, 0.2); color: #94a3b8; }}

        .judge-justification {{
            font-size: 0.82rem;
            color: #d1d5db;
            background: rgba(0, 0, 0, 0.25);
            padding: 0.65rem 0.85rem;
            border-radius: 8px;
            border-left: 3px solid #8b5cf6;
            line-height: 1.45;
        }}

        /* Meta-Evaluation Control Box */
        .meta-eval-box {{
            background: rgba(0, 0, 0, 0.35);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 0.6rem 0.75rem;
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
            margin-top: 0.25rem;
        }}

        .meta-eval-header {{
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: #c4b5fd;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .meta-btn-group {{
            display: flex;
            gap: 0.4rem;
        }}

        .meta-btn {{
            flex: 1;
            padding: 0.35rem 0.6rem;
            border-radius: 6px;
            border: 1px solid var(--border-color);
            background: var(--bg-card);
            color: var(--text-muted);
            font-size: 0.75rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.3rem;
        }}

        .meta-btn:hover {{
            background: rgba(255, 255, 255, 0.1);
            color: white;
        }}

        .meta-btn.active-correct {{
            background: rgba(16, 185, 129, 0.25);
            border-color: #10b981;
            color: #34d399;
            box-shadow: 0 0 8px rgba(16, 185, 129, 0.3);
        }}

        .meta-btn.active-incorrect {{
            background: rgba(239, 68, 68, 0.25);
            border-color: #ef4444;
            color: #f87171;
            box-shadow: 0 0 8px rgba(239, 68, 68, 0.3);
        }}

        .meta-note-input {{
            width: 100%;
            background: var(--bg-main);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            padding: 0.35rem 0.6rem;
            color: var(--text-main);
            font-size: 0.75rem;
            outline: none;
        }}

        .meta-note-input:focus {{
            border-color: #8b5cf6;
        }}

        /* --- HEAD TO HEAD MATRIX VIEW --- */
        .matrix-view {{
            display: none;
            flex-direction: column;
            gap: 1.5rem;
        }}

        .matrix-selector {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 14px;
            padding: 1.25rem;
            display: flex;
            align-items: center;
            gap: 1rem;
            flex-wrap: wrap;
        }}

        .question-dropdown {{
            flex: 1;
            min-width: 320px;
        }}

        .h2h-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.25rem;
        }}

        .h2h-card {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 14px;
            padding: 1.1rem;
            display: flex;
            flex-direction: column;
            gap: 0.85rem;
        }}

        .h2h-card.highlight {{
            border-color: #8b5cf6;
            background: linear-gradient(180deg, rgba(139, 92, 246, 0.08) 0%, rgba(17, 24, 39, 1) 100%);
        }}

        .copy-btn {{
            background: transparent;
            border: 1px solid var(--border-color);
            color: var(--text-muted);
            border-radius: 4px;
            padding: 2px 8px;
            font-size: 0.72rem;
            cursor: pointer;
            transition: all 0.2s;
        }}

        .copy-btn:hover {{
            color: white;
            border-color: var(--text-muted);
        }}

        .empty-state {{
            text-align: center;
            padding: 4rem 2rem;
            color: var(--text-muted);
        }}

        .pagination {{
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 0.5rem;
            margin-top: 2rem;
        }}

        .page-btn {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            color: var(--text-main);
            padding: 0.5rem 1rem;
            border-radius: 8px;
            font-weight: 600;
            font-size: 0.85rem;
            cursor: pointer;
            transition: all 0.2s;
        }}

        .page-btn:hover:not(:disabled) {{
            background: var(--bg-input);
            border-color: #8b5cf6;
        }}

        .page-btn:disabled {{
            opacity: 0.4;
            cursor: not-allowed;
        }}

        .page-info {{
            font-size: 0.85rem;
            color: var(--text-muted);
            font-weight: 500;
        }}
    </style>
</head>
<body>

    <header>
        <div class="header-container">
            <div class="brand">
                <div class="brand-icon">G</div>
                <div class="brand-text">
                    <h1>GovBench-BR • Dashboard de Avaliação & Juízes</h1>
                    <p>Métricas Léxicas, BERTScore, Consenso entre Juízes e Meta-Avaliação Humana</p>
                </div>
            </div>
            <div class="header-actions">
                <div class="audit-badge" id="audit-stats-badge">
                    <span>⚖️ Meta-Auditoria:</span>
                    <strong id="audit-count">0 juízes avaliados</strong>
                </div>
                <button class="btn-export" onclick="exportMetaEvaluations()">
                    📥 Exportar Auditoria (.JSON)
                </button>
                <div class="view-mode-tabs">
                    <button class="tab-btn active" id="tab-cards-btn" onclick="switchView('cards')">
                        📋 Visualização em Lista
                    </button>
                    <button class="tab-btn" id="tab-matrix-btn" onclick="switchView('matrix')">
                        ⚔️ Head-to-Head
                    </button>
                </div>
            </div>
        </div>
    </header>

    <main>
        <!-- SECTION TITLE -->
        <div class="section-header-title">
            <span>🏆 Ranking Comparativo e Estatísticas de Consenso dos Juízes</span>
        </div>

        <!-- KPI SECTION -->
        <section class="kpi-grid" id="kpi-container">
            <!-- Cards populated by JS -->
        </section>

        <!-- FILTERS -->
        <section class="filter-section">
            <div class="filter-row">
                <div class="search-box">
                    <span class="search-icon">🔍</span>
                    <input type="text" id="search-input" class="search-input" placeholder="Buscar por ID, pergunta, resposta, justificativa do juiz ou anotação..." oninput="applyFilters()">
                </div>
                <div class="filter-group">
                    <span class="filter-label">Modelo:</span>
                    <select id="filter-model" class="select-input" onchange="applyFilters()">
                        <option value="ALL">Todos os Modelos (840)</option>
                        <option value="qwen_finetuned">✨ Qwen 3.5 9B (Fine-Tuned LoRA)</option>
                        <option value="qwen_base">Qwen 3.5 9B (Base)</option>
                        <option value="ollama/mistral-nemo:12b">Mistral-Nemo 12B</option>
                        <option value="ollama/llama3.1:8b">Llama 3.1 8B</option>
                        <option value="ollama/deepseek-r1:8b">DeepSeek-R1 8B</option>
                    </select>
                </div>
                <div class="filter-group">
                    <span class="filter-label">Domínio:</span>
                    <select id="filter-domain" class="select-input" onchange="applyFilters()">
                        <option value="ALL">Todos os Domínios</option>
                        <option value="legislacao">Legislação</option>
                        <option value="saude">Saúde</option>
                        <option value="edu">Educação</option>
                        <option value="seguranca">Segurança Pública</option>
                    </select>
                </div>
                <div class="filter-group">
                    <span class="filter-label">Dificuldade:</span>
                    <select id="filter-diff" class="select-input" onchange="applyFilters()">
                        <option value="ALL">Todos os Níveis</option>
                        <option value="factual">Factual</option>
                        <option value="conceitual">Conceitual</option>
                        <option value="aplicado">Aplicado</option>
                    </select>
                </div>
                <div class="filter-group">
                    <span class="filter-label">Consenso dos Juízes:</span>
                    <select id="filter-consensus" class="select-input" onchange="applyFilters()">
                        <option value="ALL">Todos os Itens</option>
                        <option value="both_correct">🤝 Ambos Juízes Aprovaram (Correto)</option>
                        <option value="both_incorrect">❌ Ambos Juízes Rejeitaram</option>
                        <option value="disagree">⚡ Discordância entre Juízes</option>
                        <option value="any_hallucination">🚨 Alucinação Detectada (≥1 Juiz)</option>
                        <option value="meta_evaluated">✍️ Já Auditados por Humano</option>
                    </select>
                </div>
                <div class="filter-group">
                    <span class="filter-label">Ordenar:</span>
                    <select id="sort-by" class="select-input" onchange="applyFilters()">
                        <option value="default">Ordem Padrão (ID)</option>
                        <option value="rouge_desc">ROUGE-L (Maior → Menor)</option>
                        <option value="token_f1_desc">Token F1 (Maior → Menor)</option>
                        <option value="bertscore_desc">BERTScore (Maior → Menor)</option>
                        <option value="phi4_desc">Nota Phi-4 (Maior → Menor)</option>
                        <option value="command_desc">Nota Command-R (Maior → Menor)</option>
                        <option value="latency_desc">Latência (Mais Lento)</option>
                    </select>
                </div>
                <div class="filter-stats" id="filter-stats-text">
                    Exibindo 840 de 840 itens
                </div>
            </div>
        </section>

        <!-- LIST / CARDS VIEW -->
        <section id="cards-view-wrapper">
            <div class="items-container" id="items-list">
                <!-- Populated by JS -->
            </div>
            <div class="pagination" id="pagination-controls">
                <button class="page-btn" id="prev-page-btn" onclick="changePage(-1)">← Anterior</button>
                <span class="page-info" id="page-info-text">Página 1 de 1</span>
                <button class="page-btn" id="next-page-btn" onclick="changePage(1)">Próxima →</button>
            </div>
        </section>

        <!-- HEAD TO HEAD COMPARATOR VIEW -->
        <section class="matrix-view" id="matrix-view-wrapper">
            <div class="matrix-selector">
                <div class="filter-label">Selecione uma Pergunta para Comparação Direta:</div>
                <select id="matrix-question-select" class="select-input question-dropdown" onchange="renderMatrixQuestion()">
                    <!-- Options populated by JS -->
                </select>
            </div>

            <div class="item-card" id="matrix-question-details">
                <!-- Header, Question, Reference Answer -->
            </div>

            <div class="h2h-grid" id="matrix-cards-grid">
                <!-- 5 Model columns populated by JS -->
            </div>
        </section>
    </main>

    <script>
        const DATA = {json_data_str};
        const ITEMS = DATA.items || [];
        const SUMMARY = DATA.summary || {{}};
        const STRATA = DATA.strata || {{}};

        // LocalStorage Meta-Evaluations Storage Key
        const META_STORAGE_KEY = 'govbench_judge_meta_evaluations_v1';
        let metaEvaluations = JSON.parse(localStorage.getItem(META_STORAGE_KEY) || '{{}}');

        let currentView = 'cards';
        let filteredItems = [...ITEMS];
        let currentPage = 1;
        const pageSize = 20;

        function saveMetaEvaluations() {{
            localStorage.setItem(META_STORAGE_KEY, JSON.stringify(metaEvaluations));
            updateAuditBadge();
        }}

        function updateAuditBadge() {{
            const total = Object.keys(metaEvaluations).length;
            let thumbsUp = 0, thumbsDown = 0;
            Object.values(metaEvaluations).forEach(e => {{
                if (e.verdict === 'correct') thumbsUp++;
                if (e.verdict === 'incorrect') thumbsDown++;
            }});
            document.getElementById('audit-count').innerText = `${{total}} avaliações (${{thumbsUp}} 👍 / ${{thumbsDown}} 👎)`;
        }}

        function setJudgeVerdict(itemId, modelKey, judgeKey, verdict) {{
            const key = `${{itemId}}__${{modelKey}}__${{judgeKey}}`;
            if (!metaEvaluations[key]) {{
                metaEvaluations[key] = {{ itemId, model: modelKey, judge: judgeKey, note: '', timestamp: new Date().toISOString() }};
            }}
            if (metaEvaluations[key].verdict === verdict) {{
                delete metaEvaluations[key]; // toggle off
            }} else {{
                metaEvaluations[key].verdict = verdict;
                metaEvaluations[key].timestamp = new Date().toISOString();
            }}
            saveMetaEvaluations();
            renderList();
        }}

        function setJudgeNote(itemId, modelKey, judgeKey, note) {{
            const key = `${{itemId}}__${{modelKey}}__${{judgeKey}}`;
            if (!metaEvaluations[key]) {{
                metaEvaluations[key] = {{ itemId, model: modelKey, judge: judgeKey, verdict: null, timestamp: new Date().toISOString() }};
            }}
            metaEvaluations[key].note = note;
            saveMetaEvaluations();
        }}

        function exportMetaEvaluations() {{
            const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(metaEvaluations, null, 2));
            const downloadAnchor = document.createElement('a');
            downloadAnchor.setAttribute("href", dataStr);
            downloadAnchor.setAttribute("download", "govbench_judge_meta_evaluations.json");
            document.body.appendChild(downloadAnchor);
            downloadAnchor.click();
            downloadAnchor.remove();
        }}

        function init() {{
            updateAuditBadge();
            renderKPIs();
            populateMatrixSelect();
            applyFilters();
        }}

        function getModelDisplayName(key) {{
            const names = {{
                "qwen_finetuned": "✨ Qwen 3.5 9B (Fine-Tuned)",
                "qwen_base": "Qwen 3.5 9B (Base)",
                "ollama/llama3.1:8b": "Llama 3.1 8B",
                "ollama/mistral-nemo:12b": "Mistral-Nemo 12B",
                "ollama/deepseek-r1:8b": "DeepSeek-R1 8B"
            }};
            return names[key] || key;
        }}

        function getModelClass(key) {{
            return 'model-' + key.replace('ollama/', '').replace(':', '_').replace('.', '_').replace('-', '_');
        }}

        function computeModelDetailedStats(mKey) {{
            const modelItems = ITEMS.filter(i => i.model === mKey);
            const total = modelItems.length;

            let cmd_corr = 0, cmd_comp = 0, cmd_aluc = 0;
            let phi_corr = 0, phi_comp = 0, phi_aluc = 0;
            let both_corr = 0, both_incorr = 0, disagree = 0;

            modelItems.forEach(it => {{
                const cmd = it.judge_verdicts?.['ollama/command-r7b'] || {{}};
                const phi = it.judge_verdicts?.['ollama/phi4:14b'] || {{}};

                if (cmd.correto) cmd_corr++;
                if (cmd.completo) cmd_comp++;
                if (cmd.alucinacao) cmd_aluc++;

                if (phi.correto) phi_corr++;
                if (phi.completo) phi_comp++;
                if (phi.alucinacao) phi_aluc++;

                if (cmd.correto && phi.correto) {{
                    both_corr++;
                }} else if (!cmd.correto && !phi.correto) {{
                    both_incorr++;
                }} else {{
                    disagree++;
                }}
            }});

            return {{
                total,
                cmd: {{ corr: cmd_corr, comp: cmd_comp, aluc: cmd_aluc }},
                phi: {{ corr: phi_corr, comp: phi_comp, aluc: phi_aluc }},
                both_corr,
                both_incorr,
                disagree
            }};
        }}

        function renderKPIs() {{
            const container = document.getElementById('kpi-container');
            const modelKeys = ['qwen_finetuned', 'ollama/mistral-nemo:12b', 'ollama/llama3.1:8b', 'ollama/deepseek-r1:8b', 'qwen_base'];
            const colors = {{
                'qwen_finetuned': '#8b5cf6',
                'ollama/mistral-nemo:12b': '#f59e0b',
                'ollama/llama3.1:8b': '#3b82f6',
                'ollama/deepseek-r1:8b': '#06b6d4',
                'qwen_base': '#64748b'
            }};

            let html = '';
            modelKeys.forEach(m => {{
                const s = SUMMARY[m] || {{}};
                const d = computeModelDetailedStats(m);
                const isHighlight = m === 'qwen_finetuned';
                const cardColor = colors[m] || '#64748b';

                const pBothCorr = ((d.both_corr / d.total) * 100).toFixed(1);
                const pBothIncorr = ((d.both_incorr / d.total) * 100).toFixed(1);
                const pDisagree = ((d.disagree / d.total) * 100).toFixed(1);

                html += `
                    <div class="kpi-card ${{isHighlight ? 'highlight' : ''}}" style="--card-color: ${{cardColor}};">
                        <div class="kpi-header">
                            <div class="kpi-title">
                                ${{getModelDisplayName(m)}}
                            </div>
                            <span class="kpi-badge" style="background: ${{cardColor}}25; color: ${{cardColor}};">
                                ${{isHighlight ? '🏆 1º Lugar Geral' : 'Baseline'}}
                            </span>
                        </div>

                        <div class="kpi-metrics-grid">
                            <div class="kpi-metric-item">
                                <div class="kpi-metric-label">ROUGE-L</div>
                                <div class="kpi-metric-val" style="color: ${{s.rouge_l_medio > 0.25 ? '#34d399' : '#f3f4f6'}};">
                                    ${{(s.rouge_l_medio || 0).toFixed(4)}}
                                </div>
                            </div>
                            <div class="kpi-metric-item">
                                <div class="kpi-metric-label">BERTScore</div>
                                <div class="kpi-metric-val" style="color: ${{s.bertscore_f1_medio > 0.65 ? '#a78bfa' : '#f3f4f6'}};">
                                    ${{(s.bertscore_f1_medio || 0).toFixed(4)}}
                                </div>
                            </div>
                            <div class="kpi-metric-item">
                                <div class="kpi-metric-label">Token F1</div>
                                <div class="kpi-metric-val">
                                    ${{(s.token_f1_medio || 0).toFixed(4)}}
                                </div>
                            </div>
                        </div>

                        <div class="kpi-breakdown-section">
                            <div class="breakdown-row">
                                <span class="breakdown-title">⚖️ Command-R (7B):</span>
                                <div class="breakdown-chips">
                                    <span class="b-chip b-ok" title="Corretos">✔️ ${{d.cmd.corr}}</span>
                                    <span class="b-chip b-comp" title="Completos">🎯 ${{d.cmd.comp}}</span>
                                    <span class="b-chip b-aluc" title="Alucinações">🚨 ${{d.cmd.aluc}}</span>
                                </div>
                            </div>
                            <div class="breakdown-row">
                                <span class="breakdown-title">⚖️ Phi-4 (14B):</span>
                                <div class="breakdown-chips">
                                    <span class="b-chip b-ok" title="Corretos">✔️ ${{d.phi.corr}}</span>
                                    <span class="b-chip b-comp" title="Completos">🎯 ${{d.phi.comp}}</span>
                                    <span class="b-chip b-aluc" title="Alucinações">🚨 ${{d.phi.aluc}}</span>
                                </div>
                            </div>

                            <div class="agreement-bar-container">
                                <div class="agreement-bar-label">
                                    <span>Consenso: 🤝 ${{d.both_corr}} | ❌ ${{d.both_incorr}} | ⚡ ${{d.disagree}}</span>
                                    <span>${{pBothCorr}}% concordam OK</span>
                                </div>
                                <div class="bar-track" title="Verde: Ambos Correto (${{d.both_corr}}) | Cinza: Ambos Incorreto (${{d.both_incorr}}) | Amarelo: Discordam (${{d.disagree}})">
                                    <div class="bar-seg seg-both-ok" style="width: ${{pBothCorr}}%;"></div>
                                    <div class="bar-seg seg-disagree" style="width: ${{pDisagree}}%;"></div>
                                    <div class="bar-seg seg-both-fail" style="width: ${{pBothIncorr}}%;"></div>
                                </div>
                            </div>
                        </div>
                    </div>
                `;
            }});
            container.innerHTML = html;
        }}

        function applyFilters() {{
            const search = document.getElementById('search-input').value.toLowerCase().trim();
            const modelFilter = document.getElementById('filter-model').value;
            const domainFilter = document.getElementById('filter-domain').value;
            const diffFilter = document.getElementById('filter-diff').value;
            const consensusFilter = document.getElementById('filter-consensus').value;
            const sortBy = document.getElementById('sort-by').value;

            filteredItems = ITEMS.filter(item => {{
                if (modelFilter !== 'ALL' && item.model !== modelFilter) return false;
                if (domainFilter !== 'ALL' && item.dominio !== domainFilter) return false;
                if (diffFilter !== 'ALL' && item.nivel_dificuldade !== diffFilter) return false;

                const cmd = item.judge_verdicts?.['ollama/command-r7b'] || {{}};
                const phi = item.judge_verdicts?.['ollama/phi4:14b'] || {{}};

                if (consensusFilter === 'both_correct') {{
                    if (!(cmd.correto && phi.correto)) return false;
                }} else if (consensusFilter === 'both_incorrect') {{
                    if (!(!cmd.correto && !phi.correto)) return false;
                }} else if (consensusFilter === 'disagree') {{
                    if ((cmd.correto && phi.correto) || (!cmd.correto && !phi.correto)) return false;
                }} else if (consensusFilter === 'any_hallucination') {{
                    if (!(cmd.alucinacao || phi.alucinacao)) return false;
                }} else if (consensusFilter === 'meta_evaluated') {{
                    const keyCmd = `${{item.id}}__${{item.model}}__ollama/command-r7b`;
                    const keyPhi = `${{item.id}}__${{item.model}}__ollama/phi4:14b`;
                    if (!metaEvaluations[keyCmd] && !metaEvaluations[keyPhi]) return false;
                }}

                if (search) {{
                    const matchId = (item.id || '').toLowerCase().includes(search);
                    const matchQ = (item.pergunta || '').toLowerCase().includes(search);
                    const matchRef = (item.resposta_referencia || '').toLowerCase().includes(search);
                    const matchGen = (item.resposta_gerada || '').toLowerCase().includes(search);
                    const matchJust = Object.values(item.judge_verdicts || {{}})
                        .some(j => (j.justificativa || '').toLowerCase().includes(search));
                    
                    const keyCmd = `${{item.id}}__${{item.model}}__ollama/command-r7b`;
                    const keyPhi = `${{item.id}}__${{item.model}}__ollama/phi4:14b`;
                    const matchNotes = (metaEvaluations[keyCmd]?.note || '').toLowerCase().includes(search) ||
                                       (metaEvaluations[keyPhi]?.note || '').toLowerCase().includes(search);

                    if (!matchId && !matchQ && !matchRef && !matchGen && !matchJust && !matchNotes) return false;
                }}
                return true;
            }});

            // Sorting
            filteredItems.sort((a, b) => {{
                if (sortBy === 'rouge_desc') return (b.rouge_l || 0) - (a.rouge_l || 0);
                if (sortBy === 'token_f1_desc') return (b.token_f1 || 0) - (a.token_f1 || 0);
                if (sortBy === 'bertscore_desc') return (b.bertscore_f1 || 0) - (a.bertscore_f1 || 0);
                if (sortBy === 'phi4_desc') {{
                    const na = a.judge_verdicts?.['ollama/phi4:14b']?.nota_geral || 0;
                    const nb = b.judge_verdicts?.['ollama/phi4:14b']?.nota_geral || 0;
                    return nb - na;
                }}
                if (sortBy === 'command_desc') {{
                    const na = a.judge_verdicts?.['ollama/command-r7b']?.nota_geral || 0;
                    const nb = b.judge_verdicts?.['ollama/command-r7b']?.nota_geral || 0;
                    return nb - na;
                }}
                if (sortBy === 'latency_desc') return (b.latency_s || 0) - (a.latency_s || 0);
                return a.id.localeCompare(b.id);
            }});

            currentPage = 1;
            renderList();
        }}

        function getScoreClass(val, type) {{
            if (val == null) return '';
            if (type === 'rouge' || type === 'f1') {{
                if (val >= 0.3) return 'metric-high';
                if (val >= 0.15) return 'metric-mid';
                return 'metric-low';
            }}
            if (type === 'bert') {{
                if (val >= 0.65) return 'metric-high';
                if (val >= 0.5) return 'metric-mid';
                return 'metric-low';
            }}
            return '';
        }}

        function renderList() {{
            const listEl = document.getElementById('items-list');
            const statsEl = document.getElementById('filter-stats-text');
            statsEl.innerText = `Exibindo ${{filteredItems.length}} de ${{ITEMS.length}} predições`;

            if (filteredItems.length === 0) {{
                listEl.innerHTML = '<div class="empty-state">Nenhum item encontrado com os filtros selecionados.</div>';
                document.getElementById('pagination-controls').style.display = 'none';
                return;
            }}

            document.getElementById('pagination-controls').style.display = 'flex';
            const totalPages = Math.ceil(filteredItems.length / pageSize);
            document.getElementById('page-info-text').innerText = `Página ${{currentPage}} de ${{totalPages}}`;
            document.getElementById('prev-page-btn').disabled = currentPage <= 1;
            document.getElementById('next-page-btn').disabled = currentPage >= totalPages;

            const start = (currentPage - 1) * pageSize;
            const pageItems = filteredItems.slice(start, start + pageSize);

            let html = '';
            pageItems.forEach(item => {{
                const verdicts = item.judge_verdicts || {{}};
                const cmd = verdicts['ollama/command-r7b'] || {{}};
                const phi = verdicts['ollama/phi4:14b'] || {{}};

                const keyCmd = `${{item.id}}__${{item.model}}__ollama/command-r7b`;
                const keyPhi = `${{item.id}}__${{item.model}}__ollama/phi4:14b`;
                const metaCmd = metaEvaluations[keyCmd] || {{}};
                const metaPhi = metaEvaluations[keyPhi] || {{}};

                html += `
                    <div class="item-card">
                        <div class="item-header">
                            <div class="item-meta-left">
                                <span class="item-id">${{item.id}}</span>
                                <span class="model-pill ${{getModelClass(item.model)}}">${{getModelDisplayName(item.model)}}</span>
                                <span class="dom-pill dom-${{item.dominio}}">${{item.dominio}}</span>
                                <span class="diff-pill">${{item.nivel_dificuldade}}</span>
                                <span class="diff-pill">⏱️ ${{item.latency_s || '-'}}s</span>
                            </div>
                            <div class="item-metrics-bar">
                                <div class="metric-badge ${{getScoreClass(item.rouge_l, 'rouge')}}">
                                    ROUGE-L: <span class="val">${{item.rouge_l != null ? item.rouge_l.toFixed(4) : '-'}}</span>
                                </div>
                                <div class="metric-badge ${{getScoreClass(item.token_f1, 'f1')}}">
                                    Token F1: <span class="val">${{item.token_f1 != null ? item.token_f1.toFixed(4) : '-'}}</span>
                                </div>
                                <div class="metric-badge ${{getScoreClass(item.bertscore_f1, 'bert')}}">
                                    BERTScore: <span class="val">${{item.bertscore_f1 != null ? item.bertscore_f1.toFixed(4) : '-'}}</span>
                                </div>
                            </div>
                        </div>

                        <div class="item-body">
                            <div class="content-box" style="grid-column: 1 / -1;">
                                <div class="content-label">
                                    <span>❓ Pergunta</span>
                                    <button class="copy-btn" onclick="copyCardText(this, 'question')">Copiar</button>
                                </div>
                                <div class="box-text question">${{escapeHtml(item.pergunta)}}</div>
                            </div>

                            <div class="content-box">
                                <div class="content-label">
                                    <span>🎯 Gabarito Validado (Referência)</span>
                                    <button class="copy-btn" onclick="copyCardText(this, 'reference')">Copiar</button>
                                </div>
                                <div class="box-text reference">${{escapeHtml(item.resposta_referencia)}}</div>
                            </div>

                            <div class="content-box">
                                <div class="content-label">
                                    <span>🤖 Resposta Gerada (${{getModelDisplayName(item.model)}})</span>
                                    <button class="copy-btn" onclick="copyCardText(this, 'generated')">Copiar</button>
                                </div>
                                <div class="box-text generated">${{escapeHtml(item.resposta_gerada || '(Resposta Vazia / Falha)')}}</div>
                            </div>

                            <div class="judges-section">
                                <div class="content-label">
                                    <span>⚖️ Parecer dos Juízes LLM & Auditoria Humana</span>
                                    <span style="font-weight: 500; font-size: 0.72rem; color: #c4b5fd;">(Avalie se o julgamento da IA foi justo ou incorreto)</span>
                                </div>
                                <div class="judges-grid">
                                    <!-- COMMAND-R7B -->
                                    <div class="judge-box">
                                        <div class="judge-header">
                                            <span class="judge-name">Cohere Command-R (7B)</span>
                                            <span class="judge-stars">⭐ ${{cmd.nota_geral || '-'}} / 5</span>
                                        </div>
                                        <div class="judge-flags">
                                            <span class="flag-pill ${{cmd.correto ? 'flag-ok' : 'flag-bad'}}">Correto: ${{cmd.correto ? 'Sim' : 'Não'}}</span>
                                            <span class="flag-pill ${{cmd.completo ? 'flag-ok' : 'flag-bad'}}">Completo: ${{cmd.completo ? 'Sim' : 'Não'}}</span>
                                            <span class="flag-pill ${{cmd.alucinacao ? 'flag-bad' : 'flag-ok'}}">Alucinação: ${{cmd.alucinacao ? 'Sim' : 'Não'}}</span>
                                        </div>
                                        <div class="judge-justification">"${{escapeHtml(cmd.justificativa || 'Sem justificativa')}}"</div>

                                        <!-- Meta-Evaluation Box for Command-R -->
                                        <div class="meta-eval-box">
                                            <div class="meta-eval-header">
                                                <span>Auditoria Humana do Juiz</span>
                                                <span style="font-size: 0.68rem; color: var(--text-dim);">${{metaCmd.verdict ? (metaCmd.verdict === 'correct' ? '✅ Juiz Validado' : '❌ Juiz Refutado') : 'Pendente'}}</span>
                                            </div>
                                            <div class="meta-btn-group">
                                                <button class="meta-btn ${{metaCmd.verdict === 'correct' ? 'active-correct' : ''}}" onclick="setJudgeVerdict('${{item.id}}', '${{item.model}}', 'ollama/command-r7b', 'correct')">
                                                    👍 Juiz Correto / Justo
                                                </button>
                                                <button class="meta-btn ${{metaCmd.verdict === 'incorrect' ? 'active-incorrect' : ''}}" onclick="setJudgeVerdict('${{item.id}}', '${{item.model}}', 'ollama/command-r7b', 'incorrect')">
                                                    👎 Juiz Errou / Injusto
                                                </button>
                                            </div>
                                            <input type="text" class="meta-note-input" placeholder="Nota humana sobre este julgamento (opcional)..." value="${{escapeHtml(metaCmd.note || '')}}" onchange="setJudgeNote('${{item.id}}', '${{item.model}}', 'ollama/command-r7b', this.value)">
                                        </div>
                                    </div>

                                    <!-- PHI-4 -->
                                    <div class="judge-box">
                                        <div class="judge-header">
                                            <span class="judge-name">Microsoft Phi-4 (14B)</span>
                                            <span class="judge-stars">⭐ ${{phi.nota_geral || '-'}} / 5</span>
                                        </div>
                                        <div class="judge-flags">
                                            <span class="flag-pill ${{phi.correto ? 'flag-ok' : 'flag-bad'}}">Correto: ${{phi.correto ? 'Sim' : 'Não'}}</span>
                                            <span class="flag-pill ${{phi.completo ? 'flag-ok' : 'flag-bad'}}">Completo: ${{phi.completo ? 'Sim' : 'Não'}}</span>
                                            <span class="flag-pill ${{phi.alucinacao ? 'flag-bad' : 'flag-ok'}}">Alucinação: ${{phi.alucinacao ? 'Sim' : 'Não'}}</span>
                                        </div>
                                        <div class="judge-justification">"${{escapeHtml(phi.justificativa || 'Sem justificativa')}}"</div>

                                        <!-- Meta-Evaluation Box for Phi-4 -->
                                        <div class="meta-eval-box">
                                            <div class="meta-eval-header">
                                                <span>Auditoria Humana do Juiz</span>
                                                <span style="font-size: 0.68rem; color: var(--text-dim);">${{metaPhi.verdict ? (metaPhi.verdict === 'correct' ? '✅ Juiz Validado' : '❌ Juiz Refutado') : 'Pendente'}}</span>
                                            </div>
                                            <div class="meta-btn-group">
                                                <button class="meta-btn ${{metaPhi.verdict === 'correct' ? 'active-correct' : ''}}" onclick="setJudgeVerdict('${{item.id}}', '${{item.model}}', 'ollama/phi4:14b', 'correct')">
                                                    👍 Juiz Correto / Justo
                                                </button>
                                                <button class="meta-btn ${{metaPhi.verdict === 'incorrect' ? 'active-incorrect' : ''}}" onclick="setJudgeVerdict('${{item.id}}', '${{item.model}}', 'ollama/phi4:14b', 'incorrect')">
                                                    👎 Juiz Errou / Injusto
                                                </button>
                                            </div>
                                            <input type="text" class="meta-note-input" placeholder="Nota humana sobre este julgamento (opcional)..." value="${{escapeHtml(metaPhi.note || '')}}" onchange="setJudgeNote('${{item.id}}', '${{item.model}}', 'ollama/phi4:14b', this.value)">
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                `;
            }});
            listEl.innerHTML = html;
        }}

        function changePage(delta) {{
            currentPage += delta;
            renderList();
            window.scrollTo({{ top: document.querySelector('.filter-section').offsetTop - 80, behavior: 'smooth' }});
        }}

        function switchView(mode) {{
            currentView = mode;
            document.getElementById('tab-cards-btn').classList.toggle('active', mode === 'cards');
            document.getElementById('tab-matrix-btn').classList.toggle('active', mode === 'matrix');
            document.getElementById('cards-view-wrapper').style.display = mode === 'cards' ? 'block' : 'none';
            document.getElementById('matrix-view-wrapper').style.display = mode === 'matrix' ? 'flex' : 'none';
            if (mode === 'matrix') {{
                renderMatrixQuestion();
            }}
        }}

        function populateMatrixSelect() {{
            const select = document.getElementById('matrix-question-select');
            const uniqueIds = [...new Set(ITEMS.map(i => i.id))].sort();
            let html = '';
            uniqueIds.forEach(id => {{
                const item = ITEMS.find(i => i.id === id);
                html += `<option value="${{id}}">${{id}} [${{item.dominio}} / ${{item.nivel_dificuldade}}] - ${{item.pergunta.substring(0, 80)}}...</option>`;
            }});
            select.innerHTML = html;
        }}

        function renderMatrixQuestion() {{
            const qId = document.getElementById('matrix-question-select').value;
            const modelItems = ITEMS.filter(i => i.id === qId);
            if (!modelItems.length) return;

            const baseItem = modelItems[0];
            const detailsEl = document.getElementById('matrix-question-details');

            detailsEl.innerHTML = `
                <div class="item-header">
                    <div class="item-meta-left">
                        <span class="item-id">${{baseItem.id}}</span>
                        <span class="dom-pill dom-${{baseItem.dominio}}">${{baseItem.dominio}}</span>
                        <span class="diff-pill">${{baseItem.nivel_dificuldade}}</span>
                    </div>
                </div>
                <div style="padding: 1.25rem; display: flex; flex-direction: column; gap: 1rem;">
                    <div class="content-box">
                        <div class="content-label"><span>❓ Pergunta</span></div>
                        <div class="box-text question">${{escapeHtml(baseItem.pergunta)}}</div>
                    </div>
                    <div class="content-box">
                        <div class="content-label"><span>🎯 Gabarito Validado (Referência)</span></div>
                        <div class="box-text reference">${{escapeHtml(baseItem.resposta_referencia)}}</div>
                    </div>
                </div>
            `;

            const order = ['qwen_finetuned', 'ollama/mistral-nemo:12b', 'ollama/llama3.1:8b', 'ollama/deepseek-r1:8b', 'qwen_base'];
            const gridEl = document.getElementById('matrix-cards-grid');

            let gridHtml = '';
            order.forEach(mKey => {{
                const item = modelItems.find(i => i.model === mKey);
                if (!item) return;

                const isHighlight = mKey === 'qwen_finetuned';
                const cmd = item.judge_verdicts?.['ollama/command-r7b'] || {{}};
                const phi = item.judge_verdicts?.['ollama/phi4:14b'] || {{}};

                gridHtml += `
                    <div class="h2h-card ${{isHighlight ? 'highlight' : ''}}">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span class="model-pill ${{getModelClass(mKey)}}">${{getModelDisplayName(mKey)}}</span>
                            <span style="font-size: 0.75rem; color: var(--text-muted);">⏱️ ${{item.latency_s || '-'}}s</span>
                        </div>

                        <div style="display: flex; gap: 0.4rem; flex-wrap: wrap;">
                            <span class="metric-badge ${{getScoreClass(item.rouge_l, 'rouge')}}">R-L: ${{item.rouge_l != null ? item.rouge_l.toFixed(3) : '-'}}</span>
                            <span class="metric-badge ${{getScoreClass(item.bertscore_f1, 'bert')}}">BERT: ${{item.bertscore_f1 != null ? item.bertscore_f1.toFixed(3) : '-'}}</span>
                        </div>

                        <div class="box-text generated" style="max-height: 240px; overflow-y: auto; font-size: 0.82rem;">
                            ${{escapeHtml(item.resposta_gerada || '(Vazio)')}}
                        </div>

                        <div style="border-top: 1px dashed var(--border-color); padding-top: 0.5rem; display: flex; flex-direction: column; gap: 0.35rem; font-size: 0.76rem;">
                            <div style="display: flex; justify-content: space-between;">
                                <span>Command-R: ⭐ ${{cmd.nota_geral || '-'}}/5</span>
                                <span class="flag-pill ${{cmd.correto ? 'flag-ok' : 'flag-bad'}}">${{cmd.correto ? 'Correto' : 'Incorreto'}}</span>
                            </div>
                            <div style="display: flex; justify-content: space-between;">
                                <span>Phi-4: ⭐ ${{phi.nota_geral || '-'}}/5</span>
                                <span class="flag-pill ${{phi.correto ? 'flag-ok' : 'flag-bad'}}">${{phi.correto ? 'Correto' : 'Incorreto'}}</span>
                            </div>
                        </div>
                    </div>
                `;
            }});
            gridEl.innerHTML = gridHtml;
        }}

        function copyCardText(btn, type) {{
            const card = btn.closest('.item-card');
            const targetEl = card ? card.querySelector(`.box-text.${{type}}`) : null;
            if (targetEl) {{
                navigator.clipboard.writeText(targetEl.innerText || '');
                const oldText = btn.innerText;
                btn.innerText = 'Copiado!';
                btn.style.color = '#34d399';
                btn.style.borderColor = '#34d399';
                setTimeout(() => {{
                    btn.innerText = oldText;
                    btn.style.color = '';
                    btn.style.borderColor = '';
                }}, 1500);
            }}
        }}

        function escapeHtml(text) {{
            if (!text) return '';
            return text
                .replace(/&/g, "&amp;")
                .replace(/</g, "&lt;")
                .replace(/>/g, "&gt;")
                .replace(/"/g, "&quot;")
                .replace(/'/g, "&#039;");
        }}

        window.onload = init;
    </script>
</body>
</html>
"""

    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Visualizador atualizado com sucesso em: {out_html}")

if __name__ == "__main__":
    build_eval_viewer()
