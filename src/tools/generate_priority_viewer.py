#!/usr/bin/env python3
"""
generate_priority_viewer.py
----------------------------
Gera uma interface HTML interativa e um JSON detalhado para a inspeção humana
dos itens em `priority_review.jsonl`.

USO:
    python src/tools/generate_priority_viewer.py
"""

import argparse
import json
import os
from pathlib import Path

def load_jsonl(path: Path) -> list:
    if not path.exists():
        print(f"Aviso: Arquivo não encontrado: {path}")
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def compute_item_reason(judge_verdicts: dict) -> str:
    verdicts = {k: v.get("veredito") for k, v in judge_verdicts.items() if isinstance(v, dict)}
    unique_verdicts = set(verdicts.values())
    
    # Se ambos rejeitaram
    if all(v == "rejeitar" for v in verdicts.values()) and len(verdicts) > 0:
        return "ambos_rejeitar"
    # Se houve discordância entre os juízes
    if len(verdicts) > 1 and len(unique_verdicts) > 1:
        return "discordancia"
    # Se apenas command-r7b rejeitou
    for judge, v in verdicts.items():
        if v in ("rejeitar", "revisar") and "command" in judge.lower():
            return "rejeitado_command"
        if v in ("rejeitar", "revisar") and "phi" in judge.lower():
            return "rejeitado_phi4"
    
    return "outros"

def main():
    parser = argparse.ArgumentParser(description="Gera visualizador HTML para priority_review.jsonl")
    parser.add_argument(
        "--priority-file",
        default="out/06_llm_judge_out/priority_review.jsonl",
        help="Caminho para priority_review.jsonl"
    )
    parser.add_argument(
        "--clean-file",
        default="out/05_cleaned_dataset_out/govbench_br_raw_gemma4-31b_clean.jsonl",
        help="Caminho para o dataset limpo com perguntas e trechos"
    )
    parser.add_argument(
        "--output-dir",
        default="out/06_llm_judge_out",
        help="Diretório de saída para salvar o HTML e o JSON formatado"
    )
    args = parser.parse_args()

    priority_path = Path(args.priority_file)
    clean_path = Path(args.clean_file)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    priority_items = load_jsonl(priority_path)
    clean_items = {item["id"]: item for item in load_jsonl(clean_path)}

    enriched_items = []
    for item in priority_items:
        item_id = item["id"]
        clean_info = clean_items.get(item_id, {})
        
        chunk_texto = clean_info.get("chunk_texto", "")
        if isinstance(chunk_texto, list):
            chunk_texto = "\n\n---\n\n".join(chunk_texto)

        reason = compute_item_reason(item.get("judge_verdicts", {}))

        enriched = {
            "id": item_id,
            "dominio": item.get("dominio", clean_info.get("dominio", "desconhecido")),
            "nivel_dificuldade": item.get("nivel_dificuldade", clean_info.get("nivel_dificuldade", "desconhecido")),
            "pergunta": clean_info.get("pergunta", "N/A"),
            "resposta_referencia": clean_info.get("resposta_referencia", "N/A"),
            "trechos_usados": clean_info.get("trechos_usados", []),
            "fontes": clean_info.get("fontes", []),
            "chunk_texto": chunk_texto,
            "judge_verdicts": item.get("judge_verdicts", {}),
            "reason": reason
        }
        enriched_items.append(enriched)

    # 1. Salva JSON estruturado e formatado
    detailed_json_path = out_dir / "priority_review_detailed.json"
    with open(detailed_json_path, "w", encoding="utf-8") as f:
        json.dump(enriched_items, f, ensure_ascii=False, indent=2)
    print(f"[OK] JSON detalhado salvo em: {detailed_json_path}")

    # 2. Salva Dashboard HTML interativo
    html_content = generate_html_dashboard(enriched_items)
    viewer_html_path = out_dir / "priority_review_viewer.html"
    with open(viewer_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"[OK] Visualizador HTML interativo salvo em: {viewer_html_path}")


def generate_html_dashboard(items: list) -> str:
    json_str = json.dumps(items, ensure_ascii=False)
    
    html = f"""<!DOCTYPE html>
<html lang="pt-BR" data-theme="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GovBench-BR | Dashboard de Revisão de Prioridade</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-main: #0f172a;
            --bg-card: #1e293b;
            --bg-card-hover: #334155;
            --bg-header: #090d16;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --border-color: #334155;
            --accent-blue: #38bdf8;
            --accent-purple: #c084fc;
            --badge-green-bg: #064e3b;
            --badge-green-txt: #34d399;
            --badge-red-bg: #7f1d1d;
            --badge-red-txt: #f87171;
            --badge-amber-bg: #78350f;
            --badge-amber-txt: #fbbf24;
            --badge-blue-bg: #1e3a8a;
            --badge-blue-txt: #60a5fa;
            --shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5), 0 8px 10px -6px rgba(0, 0, 0, 0.4);
            --radius: 12px;
        }}

        [data-theme="light"] {{
            --bg-main: #f8fafc;
            --bg-card: #ffffff;
            --bg-card-hover: #f1f5f9;
            --bg-header: #ffffff;
            --text-primary: #0f172a;
            --text-secondary: #475569;
            --text-muted: #94a3b8;
            --border-color: #e2e8f0;
            --accent-blue: #0284c7;
            --accent-purple: #9333ea;
            --badge-green-bg: #d1fae5;
            --badge-green-txt: #047857;
            --badge-red-bg: #fee2e2;
            --badge-red-txt: #b91c1c;
            --badge-amber-bg: #fef3c7;
            --badge-amber-txt: #b45309;
            --badge-blue-bg: #dbeafe;
            --badge-blue-txt: #1d4ed8;
            --shadow: 0 10px 15px -3px rgba(0,0,0,0.08);
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }}

        body {{
            background-color: var(--bg-main);
            color: var(--text-primary);
            line-height: 1.5;
            padding-bottom: 60px;
            transition: background-color 0.2s, color 0.2s;
        }}

        header {{
            background-color: var(--bg-header);
            border-bottom: 1px solid var(--border-color);
            padding: 20px 40px;
            position: sticky;
            top: 0;
            z-index: 100;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }}

        .header-container {{
            max-width: 1400px;
            margin: 0 auto;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 15px;
        }}

        .brand {{
            display: flex;
            align-items: center;
            gap: 12px;
        }}

        .brand-logo {{
            background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
            color: white;
            font-weight: 800;
            font-size: 1.2rem;
            padding: 8px 14px;
            border-radius: 8px;
            letter-spacing: 0.5px;
        }}

        .brand-title h1 {{
            font-size: 1.3rem;
            font-weight: 700;
            color: var(--text-primary);
        }}

        .brand-title p {{
            font-size: 0.82rem;
            color: var(--text-secondary);
        }}

        .header-actions {{
            display: flex;
            align-items: center;
            gap: 12px;
        }}

        .btn {{
            background-color: var(--bg-card);
            color: var(--text-primary);
            border: 1px solid var(--border-color);
            padding: 8px 16px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 0.85rem;
            font-weight: 500;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            transition: all 0.2s;
        }}

        .btn:hover {{
            background-color: var(--bg-card-hover);
            border-color: var(--text-muted);
        }}

        .btn-primary {{
            background: linear-gradient(135deg, #2563eb, #1d4ed8);
            color: white;
            border: none;
        }}

        .btn-primary:hover {{
            background: linear-gradient(135deg, #1d4ed8, #1e40af);
        }}

        /* Container Layout */
        .main-container {{
            max-width: 1400px;
            margin: 30px auto;
            padding: 0 20px;
        }}

        /* Summary Stats Cards */
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
            margin-bottom: 25px;
        }}

        .stat-card {{
            background-color: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: var(--radius);
            padding: 18px 22px;
            display: flex;
            flex-direction: column;
            box-shadow: var(--shadow);
        }}

        .stat-card .label {{
            font-size: 0.8rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-weight: 600;
        }}

        .stat-card .value {{
            font-size: 1.8rem;
            font-weight: 700;
            margin-top: 6px;
            color: var(--text-primary);
        }}

        .stat-card .subtext {{
            font-size: 0.75rem;
            color: var(--text-muted);
            margin-top: 4px;
        }}

        /* Filter Controls */
        .controls-card {{
            background-color: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: var(--radius);
            padding: 20px;
            margin-bottom: 25px;
            box-shadow: var(--shadow);
            display: flex;
            flex-direction: column;
            gap: 16px;
        }}

        .search-box {{
            width: 100%;
            position: relative;
        }}

        .search-input {{
            width: 100%;
            padding: 12px 18px 12px 42px;
            border-radius: 8px;
            border: 1px solid var(--border-color);
            background-color: var(--bg-main);
            color: var(--text-primary);
            font-size: 0.95rem;
            outline: none;
            transition: border-color 0.2s;
        }}

        .search-input:focus {{
            border-color: var(--accent-blue);
        }}

        .search-icon {{
            position: absolute;
            left: 14px;
            top: 50%;
            transform: translateY(-50%);
            color: var(--text-muted);
            font-size: 1rem;
        }}

        .filters-group {{
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            align-items: center;
        }}

        .filter-section {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .filter-label {{
            font-size: 0.82rem;
            font-weight: 600;
            color: var(--text-secondary);
            white-space: nowrap;
        }}

        .pill-group {{
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
        }}

        .pill {{
            background-color: var(--bg-main);
            border: 1px solid var(--border-color);
            color: var(--text-secondary);
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 500;
            cursor: pointer;
            user-select: none;
            transition: all 0.2s;
        }}

        .pill:hover {{
            color: var(--text-primary);
            border-color: var(--text-muted);
        }}

        .pill.active {{
            background-color: var(--accent-blue);
            color: #0f172a;
            border-color: var(--accent-blue);
            font-weight: 600;
        }}

        /* Items List */
        .items-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding: 0 5px;
        }}

        .items-count {{
            font-size: 0.9rem;
            color: var(--text-secondary);
            font-weight: 500;
        }}

        .item-card {{
            background-color: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: var(--radius);
            margin-bottom: 20px;
            box-shadow: var(--shadow);
            overflow: hidden;
            transition: border-color 0.2s;
        }}

        .item-card:hover {{
            border-color: var(--text-muted);
        }}

        .item-header {{
            background-color: rgba(0,0,0,0.15);
            padding: 14px 20px;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 10px;
        }}

        .item-meta {{
            display: flex;
            align-items: center;
            gap: 10px;
            flex-wrap: wrap;
        }}

        .item-id {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.95rem;
            font-weight: 600;
            color: var(--accent-blue);
        }}

        .badge {{
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        .badge-domain {{
            background-color: var(--badge-blue-bg);
            color: var(--badge-blue-txt);
        }}

        .badge-level {{
            background-color: rgba(148, 163, 184, 0.15);
            color: var(--text-secondary);
            border: 1px solid var(--border-color);
        }}

        .badge-reason {{
            background-color: var(--badge-amber-bg);
            color: var(--badge-amber-txt);
        }}

        .badge-reason-rejection {{
            background-color: var(--badge-red-bg);
            color: var(--badge-red-txt);
        }}

        .badge-green {{
            background-color: var(--badge-green-bg);
            color: var(--badge-green-txt);
        }}

        .badge-red {{
            background-color: var(--badge-red-bg);
            color: var(--badge-red-txt);
        }}

        .badge-amber {{
            background-color: var(--badge-amber-bg);
            color: var(--badge-amber-txt);
        }}

        .item-content {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0;
        }}

        @media (max-width: 900px) {{
            .item-content {{
                grid-template-columns: 1fr;
            }}
        }}

        .pane-left {{
            padding: 20px;
            border-right: 1px solid var(--border-color);
            display: flex;
            flex-direction: column;
            gap: 16px;
        }}

        @media (max-width: 900px) {{
            .pane-left {{
                border-right: none;
                border-bottom: 1px solid var(--border-color);
            }}
        }}

        .pane-right {{
            padding: 20px;
            background-color: rgba(0,0,0,0.08);
            display: flex;
            flex-direction: column;
            gap: 16px;
        }}

        .section-title {{
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            color: var(--text-secondary);
            letter-spacing: 0.5px;
            margin-bottom: 6px;
            display: flex;
            align-items: center;
            gap: 6px;
        }}

        .text-block {{
            background-color: var(--bg-main);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 14px;
            font-size: 0.9rem;
            color: var(--text-primary);
            white-space: pre-wrap;
            word-break: break-word;
        }}

        .text-chunk {{
            max-height: 220px;
            overflow-y: auto;
            font-size: 0.85rem;
            line-height: 1.6;
            color: var(--text-secondary);
            font-family: 'Inter', sans-serif;
        }}

        /* Scrollbar custom */
        .text-chunk::-webkit-scrollbar {{
            width: 6px;
        }}
        .text-chunk::-webkit-scrollbar-thumb {{
            background-color: var(--border-color);
            border-radius: 3px;
        }}

        .judge-box {{
            background-color: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 14px;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }}

        .judge-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .judge-name {{
            font-weight: 600;
            font-size: 0.9rem;
            color: var(--accent-purple);
            font-family: 'JetBrains Mono', monospace;
        }}

        .criteria-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 6px;
            margin: 4px 0;
        }}

        .criterion {{
            font-size: 0.75rem;
            display: flex;
            align-items: center;
            gap: 6px;
            color: var(--text-secondary);
        }}

        .c-icon {{
            font-weight: bold;
            font-size: 0.85rem;
        }}

        .c-icon.pass {{ color: var(--badge-green-txt); }}
        .c-icon.fail {{ color: var(--badge-red-txt); }}

        .justification {{
            font-size: 0.85rem;
            background-color: var(--bg-main);
            padding: 10px 12px;
            border-radius: 6px;
            border-left: 3px solid var(--accent-purple);
            color: var(--text-primary);
            line-height: 1.45;
        }}

        .human-action-bar {{
            background-color: rgba(0,0,0,0.2);
            border-top: 1px solid var(--border-color);
            padding: 12px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 10px;
        }}

        .human-btn-group {{
            display: flex;
            gap: 8px;
        }}

        .h-btn {{
            padding: 6px 14px;
            border-radius: 6px;
            font-size: 0.8rem;
            font-weight: 600;
            cursor: pointer;
            border: 1px solid var(--border-color);
            background-color: var(--bg-main);
            color: var(--text-secondary);
            transition: all 0.2s;
        }}

        .h-btn.approve.selected {{
            background-color: var(--badge-green-bg);
            color: var(--badge-green-txt);
            border-color: var(--badge-green-txt);
        }}

        .h-btn.modify.selected {{
            background-color: var(--badge-amber-bg);
            color: var(--badge-amber-txt);
            border-color: var(--badge-amber-txt);
        }}

        .h-btn.reject.selected {{
            background-color: var(--badge-red-bg);
            color: var(--badge-red-txt);
            border-color: var(--badge-red-txt);
        }}

        .notes-input {{
            flex: 1;
            min-width: 250px;
            padding: 6px 12px;
            border-radius: 6px;
            border: 1px solid var(--border-color);
            background-color: var(--bg-main);
            color: var(--text-primary);
            font-size: 0.82rem;
            outline: none;
        }}

        .notes-input:focus {{
            border-color: var(--accent-blue);
        }}

        .no-results {{
            text-align: center;
            padding: 50px;
            color: var(--text-muted);
            font-size: 1.1rem;
        }}
    </style>
</head>
<body>

<header>
    <div class="header-container">
        <div class="brand">
            <div class="brand-logo">GovBench</div>
            <div class="brand-title">
                <h1>Painel de Revisão de Prioridade</h1>
                <p>Auditoria Camada 2 — Juízes LLM (Command-R7B vs Phi-4 14B)</p>
            </div>
        </div>
        <div class="header-actions">
            <button class="btn" id="toggleTheme">🌙 / ☀️ Tema</button>
            <button class="btn btn-primary" id="exportDecisions">📥 Exportar Decisões (.JSON)</button>
        </div>
    </div>
</header>

<div class="main-container">
    <!-- Stat Cards -->
    <div class="stats-grid">
        <div class="stat-card">
            <span class="label">Total Prioritário</span>
            <span class="value" id="statTotal">0</span>
            <span class="subtext">Itens que exigem auditoria</span>
        </div>
        <div class="stat-card">
            <span class="label">Discordâncias</span>
            <span class="value" style="color: var(--badge-amber-txt);" id="statDiscord">0</span>
            <span class="subtext">Juízes divergiram no veredito</span>
        </div>
        <div class="stat-card">
            <span class="label">Rejeitados (Command-R7B)</span>
            <span class="value" style="color: var(--badge-red-txt);" id="statCommandRej">0</span>
            <span class="subtext">Reprovados por Command-R7B</span>
        </div>
        <div class="stat-card">
            <span class="label">Rejeitados (Phi-4 14B)</span>
            <span class="value" style="color: var(--badge-red-txt);" id="statPhiRej">0</span>
            <span class="subtext">Reprovados por Phi-4 14B</span>
        </div>
    </div>

    <!-- Filter & Search Controls -->
    <div class="controls-card">
        <div class="search-box">
            <span class="search-icon">🔍</span>
            <input type="text" id="searchInput" class="search-input" placeholder="Buscar por ID, trecho da pergunta, resposta ou justificativa...">
        </div>
        <div class="filters-group">
            <div class="filter-section">
                <span class="filter-label">Domínio:</span>
                <div class="pill-group" id="filterDomain">
                    <div class="pill active" data-value="all">Todos</div>
                    <div class="pill" data-value="legislacao">Legislação</div>
                    <div class="pill" data-value="saude">Saúde</div>
                    <div class="pill" data-value="edu">Educação</div>
                    <div class="pill" data-value="seguranca">Segurança</div>
                </div>
            </div>
            <div class="filter-section">
                <span class="filter-label">Dificuldade:</span>
                <div class="pill-group" id="filterLevel">
                    <div class="pill active" data-value="all">Todos</div>
                    <div class="pill" data-value="factual">Factual</div>
                    <div class="pill" data-value="conceitual">Conceitual</div>
                    <div class="pill" data-value="aplicado">Aplicado</div>
                </div>
            </div>
            <div class="filter-section">
                <span class="filter-label">Motivo:</span>
                <div class="pill-group" id="filterReason">
                    <div class="pill active" data-value="all">Todos</div>
                    <div class="pill" data-value="discordancia">Discordância</div>
                    <div class="pill" data-value="rejeitado_command">Rejeitado Cmd-R7B</div>
                    <div class="pill" data-value="rejeitado_phi4">Rejeitado Phi-4</div>
                    <div class="pill" data-value="ambos_rejeitar">Ambos Rejeitaram</div>
                </div>
            </div>
        </div>
    </div>

    <!-- Items Feed Header -->
    <div class="items-header">
        <div class="items-count" id="itemsCountText">Exibindo 0 de 0 itens</div>
    </div>

    <!-- Container do Feed -->
    <div id="itemsContainer"></div>
</div>

<script id="dataset-data" type="application/json">
{json_str}
</script>

<script>
    const itemsData = JSON.parse(document.getElementById('dataset-data').textContent);
    let userAuditDecisions = JSON.parse(localStorage.getItem('govbench_audit_decisions') || '{{}}');

    let currentFilters = {{
        search: '',
        domain: 'all',
        level: 'all',
        reason: 'all'
    }};

    // Init Page Stats
    function initStats() {{
        document.getElementById('statTotal').textContent = itemsData.length;
        
        let discord = 0;
        let cmdRej = 0;
        let phiRej = 0;

        itemsData.forEach(item => {{
            const reason = item.reason;
            if (reason === 'discordancia') discord++;
            if (reason === 'rejeitado_command' || reason === 'ambos_rejeitar') cmdRej++;
            if (reason === 'rejeitado_phi4' || reason === 'ambos_rejeitar') phiRej++;
        }});

        document.getElementById('statDiscord').textContent = discord;
        document.getElementById('statCommandRej').textContent = cmdRej;
        document.getElementById('statPhiRej').textContent = phiRej;
    }}

    // Filter Helper
    function getFilteredItems() {{
        return itemsData.filter(item => {{
            // Search text match
            if (currentFilters.search) {{
                const q = currentFilters.search.toLowerCase();
                const matchId = item.id.toLowerCase().includes(q);
                const matchPergunta = item.pergunta.toLowerCase().includes(q);
                const matchResposta = item.resposta_referencia.toLowerCase().includes(q);
                
                let matchJust = false;
                for (let j in item.judge_verdicts) {{
                    if (item.judge_verdicts[j].justificativa && item.judge_verdicts[j].justificativa.toLowerCase().includes(q)) {{
                        matchJust = true;
                        break;
                    }}
                }}

                if (!matchId && !matchPergunta && !matchResposta && !matchJust) return false;
            }}

            // Domain match
            if (currentFilters.domain !== 'all' && item.dominio !== currentFilters.domain) return false;

            // Level match
            if (currentFilters.level !== 'all' && item.nivel_dificuldade !== currentFilters.level) return false;

            // Reason match
            if (currentFilters.reason !== 'all') {{
                if (currentFilters.reason === 'discordancia' && item.reason !== 'discordancia') return false;
                if (currentFilters.reason === 'rejeitado_command' && item.reason !== 'rejeitado_command') return false;
                if (currentFilters.reason === 'rejeitado_phi4' && item.reason !== 'rejeitado_phi4') return false;
                if (currentFilters.reason === 'ambos_rejeitar' && item.reason !== 'ambos_rejeitar') return false;
            }}

            return true;
        }});
    }}

    function renderItems() {{
        const container = document.getElementById('itemsContainer');
        const filtered = getFilteredItems();

        document.getElementById('itemsCountText').textContent = `Exibindo ${{filtered.length}} de ${{itemsData.length}} itens`;

        if (filtered.length === 0) {{
            container.innerHTML = `<div class="no-results">Nenhum item encontrado com os filtros selecionados.</div>`;
            return;
        }}

        let html = '';
        filtered.forEach(item => {{
            const decision = userAuditDecisions[item.id] || {{ action: null, notes: '' }};

            let reasonBadge = '';
            if (item.reason === 'discordancia') {{
                reasonBadge = `<span class="badge badge-reason">⚠️ Juízes Divergiram</span>`;
            }} else if (item.reason === 'ambos_rejeitar') {{
                reasonBadge = `<span class="badge badge-reason-rejection">❌ Ambos Rejeitaram</span>`;
            }} else if (item.reason === 'rejeitado_command') {{
                reasonBadge = `<span class="badge badge-reason-rejection">❌ Cmd-R7B Rejeitou</span>`;
            }} else if (item.reason === 'rejeitado_phi4') {{
                reasonBadge = `<span class="badge badge-reason-rejection">❌ Phi-4 Rejeitou</span>`;
            }}

            html += `
            <div class="item-card" id="card-${{item.id}}">
                <div class="item-header">
                    <div class="item-meta">
                        <span class="item-id">${{item.id}}</span>
                        <span class="badge badge-domain">${{item.dominio}}</span>
                        <span class="badge badge-level">${{item.nivel_dificuldade}}</span>
                        ${{reasonBadge}}
                    </div>
                </div>

                <div class="item-content">
                    <!-- Esquerda: Pergunta, Resposta, Chunk Texto -->
                    <div class="pane-left">
                        <div>
                            <div class="section-title">❓ Pergunta Gerada</div>
                            <div class="text-block">${{escapeHtml(item.pergunta)}}</div>
                        </div>

                        <div>
                            <div class="section-title">💡 Resposta de Referência (Gemma 31B Teacher)</div>
                            <div class="text-block">${{escapeHtml(item.resposta_referencia)}}</div>
                        </div>

                        <div>
                            <div class="section-title">📄 Trecho-Fonte Original (${{item.fontes ? item.fontes.join(', ') : ''}})</div>
                            <div class="text-block text-chunk">${{escapeHtml(item.chunk_texto)}}</div>
                        </div>
                    </div>

                    <!-- Direita: Juízes Lado a Lado -->
                    <div class="pane-right">
                        <div class="section-title">⚖️ Avaliação dos Juízes LLM</div>
                        ${{renderJudges(item.judge_verdicts)}}
                    </div>
                </div>

                <!-- Footer: Ação Humana -->
                <div class="human-action-bar">
                    <div class="human-btn-group">
                        <button class="h-btn approve ${{decision.action === 'manter' ? 'selected' : ''}}" onclick="setDecision('${{item.id}}', 'manter')">✅ Manter Item</button>
                        <button class="h-btn modify ${{decision.action === 'ajustar' ? 'selected' : ''}}" onclick="setDecision('${{item.id}}', 'ajustar')">✏️ Precisa de Ajuste</button>
                        <button class="h-btn reject ${{decision.action === 'descartar' ? 'selected' : ''}}" onclick="setDecision('${{item.id}}', 'descartar')">🗑️ Descartar Item</button>
                    </div>
                    <input type="text" class="notes-input" placeholder="Anotação da revisão humana (opcional)..." value="${{escapeHtml(decision.notes || '')}}" onchange="setNotes('${{item.id}}', this.value)">
                </div>
            </div>
            `;
        }});

        container.innerHTML = html;
    }}

    function renderJudges(verdicts) {{
        let html = '';
        for (let judgeName in verdicts) {{
            const j = verdicts[judgeName];
            
            let vBadge = '';
            if (j.veredito === 'aprovado') {{
                vBadge = `<span class="badge badge-green">Aprovado</span>`;
            }} else if (j.veredito === 'rejeitar') {{
                vBadge = `<span class="badge badge-red">Rejeitado</span>`;
            }} else {{
                vBadge = `<span class="badge badge-amber">${{j.veredito || 'N/A'}}</span>`;
            }}

            const fTree = j.fundamentado_no_trecho ? '<span class="c-icon pass">✓</span> Fundamentado no trecho' : '<span class="c-icon fail">✗</span> Não fundamentado';
            const fExtra = !j.informacao_extra_nao_presente ? '<span class="c-icon pass">✓</span> Sem info extra externa' : '<span class="c-icon fail">✗</span> Contém info extra';
            const fDif = j.nivel_dificuldade_adequado ? '<span class="c-icon pass">✓</span> Dificuldade adequada' : '<span class="c-icon fail">✗</span> Dificuldade inadequada';
            const fPrec = j.resposta_completa_precisa ? '<span class="c-icon pass">✓</span> Resposta precisa' : '<span class="c-icon fail">✗</span> Resposta imprecisa';

            html += `
            <div class="judge-box">
                <div class="judge-header">
                    <span class="judge-name">${{judgeName}}</span>
                    ${{vBadge}}
                </div>
                <div class="criteria-grid">
                    <div class="criterion">${{fTree}}</div>
                    <div class="criterion">${{fExtra}}</div>
                    <div class="criterion">${{fDif}}</div>
                    <div class="criterion">${{fPrec}}</div>
                </div>
                <div class="justification">
                    <strong>Justificativa:</strong> ${{escapeHtml(j.justificativa || 'Sem justificativa')}}
                </div>
            </div>
            `;
        }}
        return html;
    }}

    function escapeHtml(text) {{
        if (!text) return '';
        return String(text)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }}

    function setDecision(itemId, action) {{
        if (!userAuditDecisions[itemId]) userAuditDecisions[itemId] = {{ action: null, notes: '' }};
        if (userAuditDecisions[itemId].action === action) {{
            userAuditDecisions[itemId].action = null; // Toggle off
        }} else {{
            userAuditDecisions[itemId].action = action;
        }}
        localStorage.setItem('govbench_audit_decisions', JSON.stringify(userAuditDecisions));
        renderItems();
    }}

    function setNotes(itemId, notesText) {{
        if (!userAuditDecisions[itemId]) userAuditDecisions[itemId] = {{ action: null, notes: '' }};
        userAuditDecisions[itemId].notes = notesText;
        localStorage.setItem('govbench_audit_decisions', JSON.stringify(userAuditDecisions));
    }}

    // Export JSON decisions
    document.getElementById('exportDecisions').addEventListener('click', () => {{
        const blob = new Blob([JSON.stringify(userAuditDecisions, null, 2)], {{ type: 'application/json' }});
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'govbench_human_audit_decisions.json';
        a.click();
        URL.revokeObjectURL(url);
    }});

    // Theme toggle
    document.getElementById('toggleTheme').addEventListener('click', () => {{
        const html = document.documentElement;
        const current = html.getAttribute('data-theme');
        const next = current === 'dark' ? 'light' : 'dark';
        html.setAttribute('data-theme', next);
    }});

    // Setup Filter Pill Listeners
    function setupPills(containerId, filterKey) {{
        const container = document.getElementById(containerId);
        container.querySelectorAll('.pill').forEach(pill => {{
            pill.addEventListener('click', () => {{
                container.querySelectorAll('.pill').forEach(p => p.classList.remove('active'));
                pill.classList.add('active');
                currentFilters[filterKey] = pill.getAttribute('data-value');
                renderItems();
            }});
        }});
    }}

    setupPills('filterDomain', 'domain');
    setupPills('filterLevel', 'level');
    setupPills('filterReason', 'reason');

    // Search input listener
    document.getElementById('searchInput').addEventListener('input', (e) => {{
        currentFilters.search = e.target.value;
        renderItems();
    }});

    // Run on load
    initStats();
    renderItems();
</script>

</body>
</html>
"""
    return html

if __name__ == "__main__":
    main()
