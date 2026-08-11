#!/usr/bin/env python3
"""
generate_visualizer.py
----------------------
Gera uma interface web interativa estática (standalone HTML) para exploração, 
filtragem e análise detalhada do benchmark GovBench-BR como um todo.

Lê os arquivos de dataset higienizado e splits:
  - 05_cleaned_dataset_out/govbench_br_raw_gemma4-31b_clean.jsonl
  - 07_splits_out/govbench_br_raw_gemma4-31b_clean_train.jsonl
  - 07_splits_out/govbench_br_raw_gemma4-31b_clean_test.jsonl

Gera o arquivo de saída:
  - src/tools/benchmark_explorer.html
"""

import json
from pathlib import Path

def build_visualizer():
    root_dir = Path(__file__).resolve().parent.parent.parent
    clean_path = root_dir / "05_cleaned_dataset_out" / "govbench_br_raw_gemma4-31b_clean.jsonl"
    train_path = root_dir / "07_splits_out" / "govbench_br_raw_gemma4-31b_clean_train.jsonl"
    test_path = root_dir / "07_splits_out" / "govbench_br_raw_gemma4-31b_clean_test.jsonl"
    
    out_dir = Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_html = out_dir / "benchmark_explorer.html"
    
    if not clean_path.exists():
        print(f"Erro: Dataset higienizado não encontrado em {clean_path}")
        return

    train_ids = set()
    if train_path.exists():
        with open(train_path, 'r', encoding='utf-8') as f:
            train_ids = set(json.loads(l)['id'] for l in f if l.strip())

    test_ids = set()
    if test_path.exists():
        with open(test_path, 'r', encoding='utf-8') as f:
            test_ids = set(json.loads(l)['id'] for l in f if l.strip())

    items = []
    with open(clean_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            item_id = item['id']
            if item_id in train_ids:
                item['split'] = 'treino'
            elif item_id in test_ids:
                item['split'] = 'teste'
            else:
                item['split'] = 'geral'
            items.append(item)

    print(f"Carregados {len(items)} itens para incorporar na visualização.")

    json_data_str = json.dumps(items, ensure_ascii=False)

    html_content = f"""<!DOCTYPE html>
<html lang="pt-BR" data-theme="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GovBench-BR | Visualizador do Benchmark</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-main: #0b0f19;
            --bg-card: #131b2e;
            --bg-card-hover: #1a243d;
            --bg-input: #1c2640;
            --border-color: #243152;
            --border-highlight: #3b82f6;
            --text-main: #f1f5f9;
            --text-muted: #94a3b8;
            --text-dim: #64748b;
            
            --domain-legislação: #ec4899;
            --domain-saúde: #10b981;
            --domain-educação: #3b82f6;
            --domain-segurança: #f59e0b;
            
            --diff-factual: #06b6d4;
            --diff-conceitual: #8b5cf6;
            --diff-aplicado: #f43f5e;
            
            --split-treino: #10b981;
            --split-teste: #f59e0b;

            --radius-lg: 16px;
            --radius-md: 10px;
            --radius-sm: 6px;
            --shadow-glow: 0 0 25px rgba(59, 130, 246, 0.15);
        }}

        [data-theme="light"] {{
            --bg-main: #f8fafc;
            --bg-card: #ffffff;
            --bg-card-hover: #f1f5f9;
            --bg-input: #e2e8f0;
            --border-color: #cbd5e1;
            --border-highlight: #2563eb;
            --text-main: #0f172a;
            --text-muted: #475569;
            --text-dim: #94a3b8;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: 'Plus Jakarta Sans', sans-serif;
            background-color: var(--bg-main);
            color: var(--text-main);
            line-height: 1.6;
            padding: 24px;
            min-height: 100vh;
        }}

        .container {{
            max-width: 1440px;
            margin: 0 auto;
        }}

        /* Header */
        header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 24px;
            padding-bottom: 20px;
            border-bottom: 1px solid var(--border-color);
            flex-wrap: wrap;
            gap: 16px;
        }}

        .brand {{
            display: flex;
            align-items: center;
            gap: 12px;
        }}

        .brand-badge {{
            background: linear-gradient(135deg, #3b82f6, #8b5cf6);
            color: white;
            font-weight: 800;
            font-size: 14px;
            padding: 6px 12px;
            border-radius: var(--radius-sm);
            letter-spacing: 0.5px;
        }}

        .brand h1 {{
            font-size: 24px;
            font-weight: 800;
            letter-spacing: -0.5px;
            background: linear-gradient(135deg, #f1f5f9 0%, #94a3b8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}

        .header-actions {{
            display: flex;
            align-items: center;
            gap: 12px;
        }}

        .btn {{
            background-color: var(--bg-card);
            color: var(--text-main);
            border: 1px solid var(--border-color);
            padding: 8px 16px;
            border-radius: var(--radius-sm);
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }}

        .btn:hover {{
            background-color: var(--bg-card-hover);
            border-color: var(--border-highlight);
        }}

        .btn-primary {{
            background: linear-gradient(135deg, #2563eb, #1d4ed8);
            color: white;
            border: none;
        }}

        .btn-primary:hover {{
            box-shadow: var(--shadow-glow);
            transform: translateY(-1px);
        }}

        /* KPI Cards Grid */
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}

        .kpi-card {{
            background-color: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: var(--radius-md);
            padding: 20px;
            position: relative;
            overflow: hidden;
            transition: transform 0.2s ease, border-color 0.2s ease;
        }}

        .kpi-card:hover {{
            transform: translateY(-2px);
            border-color: var(--border-highlight);
        }}

        .kpi-title {{
            font-size: 12px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--text-muted);
            margin-bottom: 6px;
        }}

        .kpi-value {{
            font-size: 28px;
            font-weight: 800;
            line-height: 1.2;
        }}

        .kpi-subtext {{
            font-size: 12px;
            color: var(--text-dim);
            margin-top: 6px;
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }}

        /* Charts & Analytics Section */
        .analytics-container {{
            background-color: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: var(--radius-lg);
            padding: 20px;
            margin-bottom: 24px;
        }}

        .analytics-title {{
            font-size: 16px;
            font-weight: 700;
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}

        .bar-chart-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
        }}

        .chart-box {{
            background-color: var(--bg-main);
            border: 1px solid var(--border-color);
            border-radius: var(--radius-md);
            padding: 16px;
        }}

        .chart-header {{
            font-size: 13px;
            font-weight: 700;
            color: var(--text-muted);
            margin-bottom: 12px;
        }}

        .bar-row {{
            margin-bottom: 10px;
        }}

        .bar-label {{
            display: flex;
            justify-content: space-between;
            font-size: 12px;
            font-weight: 600;
            margin-bottom: 4px;
        }}

        .bar-track {{
            background-color: var(--bg-input);
            height: 10px;
            border-radius: 5px;
            overflow: hidden;
        }}

        .bar-fill {{
            height: 100%;
            border-radius: 5px;
            transition: width 0.6s ease;
        }}

        /* Controls / Filters Section */
        .filters-card {{
            background-color: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: var(--radius-lg);
            padding: 20px;
            margin-bottom: 24px;
            display: flex;
            flex-direction: column;
            gap: 16px;
        }}

        .search-row {{
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
        }}

        .search-input-wrapper {{
            flex: 1;
            min-width: 280px;
            position: relative;
        }}

        .search-input {{
            width: 100%;
            background-color: var(--bg-input);
            border: 1px solid var(--border-color);
            color: var(--text-main);
            padding: 12px 16px;
            border-radius: var(--radius-sm);
            font-size: 14px;
            outline: none;
            transition: border-color 0.2s ease;
        }}

        .search-input:focus {{
            border-color: var(--border-highlight);
            box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2);
        }}

        .select-input {{
            background-color: var(--bg-input);
            border: 1px solid var(--border-color);
            color: var(--text-main);
            padding: 12px 16px;
            border-radius: var(--radius-sm);
            font-size: 13px;
            font-weight: 600;
            outline: none;
            cursor: pointer;
        }}

        .filter-groups {{
            display: flex;
            flex-wrap: wrap;
            gap: 16px;
            align-items: center;
        }}

        .filter-group {{
            display: flex;
            align-items: center;
            gap: 8px;
            flex-wrap: wrap;
        }}

        .filter-group-title {{
            font-size: 12px;
            font-weight: 700;
            color: var(--text-dim);
            text-transform: uppercase;
            margin-right: 4px;
        }}

        .pill-btn {{
            background-color: var(--bg-input);
            color: var(--text-muted);
            border: 1px solid var(--border-color);
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
        }}

        .pill-btn:hover {{
            color: var(--text-main);
            border-color: var(--text-muted);
        }}

        .pill-btn.active {{
            background-color: #2563eb;
            color: white;
            border-color: #2563eb;
        }}

        .pill-btn.active-domain-legislação {{ background-color: var(--domain-legislação); border-color: var(--domain-legislação); color: white; }}
        .pill-btn.active-domain-saúde {{ background-color: var(--domain-saúde); border-color: var(--domain-saúde); color: white; }}
        .pill-btn.active-domain-educação {{ background-color: var(--domain-educação); border-color: var(--domain-educação); color: white; }}
        .pill-btn.active-domain-segurança {{ background-color: var(--domain-segurança); border-color: var(--domain-segurança); color: white; }}

        .results-meta {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
            font-size: 14px;
            color: var(--text-muted);
            font-weight: 600;
        }}

        /* Dataset Item List Grid */
        .items-list {{
            display: flex;
            flex-direction: column;
            gap: 16px;
        }}

        .item-card {{
            background-color: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: var(--radius-md);
            padding: 20px;
            transition: border-color 0.2s ease, box-shadow 0.2s ease;
        }}

        .item-card:hover {{
            border-color: var(--border-highlight);
        }}

        .item-header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 12px;
            margin-bottom: 14px;
            flex-wrap: wrap;
        }}

        .item-id {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 13px;
            font-weight: 600;
            color: var(--border-highlight);
            background-color: rgba(59, 130, 246, 0.1);
            padding: 4px 10px;
            border-radius: var(--radius-sm);
        }}

        .badge-list {{
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            align-items: center;
        }}

        .badge {{
            font-size: 11px;
            font-weight: 700;
            padding: 4px 10px;
            border-radius: 12px;
            text-transform: uppercase;
            letter-spacing: 0.3px;
        }}

        .badge-domain-legislação {{ background-color: rgba(236, 72, 153, 0.15); color: #f472b6; border: 1px solid rgba(236, 72, 153, 0.3); }}
        .badge-domain-saúde {{ background-color: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }}
        .badge-domain-educação {{ background-color: rgba(59, 130, 246, 0.15); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.3); }}
        .badge-domain-segurança {{ background-color: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }}

        .badge-diff-factual {{ background-color: rgba(6, 182, 212, 0.15); color: #22d3ee; border: 1px solid rgba(6, 182, 212, 0.3); }}
        .badge-diff-conceitual {{ background-color: rgba(139, 92, 246, 0.15); color: #a78bfa; border: 1px solid rgba(139, 92, 246, 0.3); }}
        .badge-diff-aplicado {{ background-color: rgba(244, 63, 94, 0.15); color: #fb7185; border: 1px solid rgba(244, 63, 94, 0.3); }}

        .badge-split-treino {{ background-color: rgba(16, 185, 129, 0.1); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.3); }}
        .badge-split-teste {{ background-color: rgba(245, 158, 11, 0.1); color: #f59e0b; border: 1px solid rgba(245, 158, 11, 0.3); }}

        .badge-group {{ background-color: var(--bg-input); color: var(--text-muted); border: 1px solid var(--border-color); }}

        .item-section {{
            margin-bottom: 14px;
        }}

        .item-label {{
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            color: var(--text-dim);
            margin-bottom: 6px;
            letter-spacing: 0.5px;
        }}

        .question-text {{
            font-size: 15px;
            font-weight: 700;
            color: var(--text-main);
            line-height: 1.5;
        }}

        .answer-box {{
            background-color: var(--bg-main);
            border: 1px solid var(--border-color);
            border-left: 4px solid var(--border-highlight);
            border-radius: var(--radius-sm);
            padding: 12px 16px;
            font-size: 14px;
            color: var(--text-main);
        }}

        .sources-chips {{
            display: flex;
            gap: 6px;
            flex-wrap: wrap;
            margin-top: 6px;
        }}

        .source-chip {{
            background-color: var(--bg-input);
            color: var(--text-muted);
            font-size: 11px;
            font-weight: 600;
            padding: 2px 8px;
            border-radius: 4px;
            font-family: 'JetBrains Mono', monospace;
        }}

        /* Context Accordion */
        details {{
            background-color: var(--bg-main);
            border: 1px solid var(--border-color);
            border-radius: var(--radius-sm);
            margin-top: 10px;
            overflow: hidden;
        }}

        summary {{
            padding: 10px 14px;
            font-size: 13px;
            font-weight: 600;
            color: var(--text-muted);
            cursor: pointer;
            user-select: none;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        summary:hover {{
            color: var(--text-main);
            background-color: var(--bg-input);
        }}

        .context-content {{
            padding: 14px;
            border-top: 1px solid var(--border-color);
            font-size: 13px;
            font-family: 'JetBrains Mono', monospace;
            white-space: pre-wrap;
            color: var(--text-muted);
            max-height: 350px;
            overflow-y: auto;
            line-height: 1.6;
        }}

        .highlight {{
            background-color: rgba(245, 158, 11, 0.3);
            color: #fef08a;
            padding: 0 2px;
            border-radius: 2px;
        }}

        /* Pagination */
        .pagination {{
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 8px;
            margin-top: 32px;
            flex-wrap: wrap;
        }}

        .page-btn {{
            min-width: 36px;
            height: 36px;
            padding: 0 10px;
            background-color: var(--bg-card);
            border: 1px solid var(--border-color);
            color: var(--text-main);
            border-radius: var(--radius-sm);
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
        }}

        .page-btn:hover {{
            background-color: var(--bg-card-hover);
            border-color: var(--border-highlight);
        }}

        .page-btn.active {{
            background-color: #2563eb;
            color: white;
            border-color: #2563eb;
        }}

        .page-btn:disabled {{
            opacity: 0.4;
            cursor: not-allowed;
        }}

        /* Modal JSON */
        .modal-overlay {{
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: rgba(0, 0, 0, 0.75);
            backdrop-filter: blur(4px);
            display: none;
            justify-content: center;
            align-items: center;
            z-index: 1000;
            padding: 20px;
        }}

        .modal-card {{
            background-color: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: var(--radius-lg);
            width: 100%;
            max-width: 800px;
            max-height: 85vh;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.5);
        }}

        .modal-header {{
            padding: 16px 20px;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .modal-body {{
            padding: 20px;
            overflow-y: auto;
            font-family: 'JetBrains Mono', monospace;
            font-size: 12px;
            color: var(--text-muted);
            white-space: pre-wrap;
        }}
    </style>
</head>
<body>

<div class="container">
    <header>
        <div class="brand">
            <span class="brand-badge">GOVBENCH-BR</span>
            <h1>Explorador e Analisador do Benchmark</h1>
        </div>
        <div class="header-actions">
            <button class="btn" onclick="toggleTheme()">🌓 Alternar Tema</button>
            <button class="btn" onclick="resetFilters()">🔄 Resetar Filtros</button>
            <button class="btn btn-primary" onclick="exportFilteredJSON()">📥 Exportar Filtrados (JSON)</button>
        </div>
    </header>

    <!-- KPI Summary Cards -->
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-title">Total de Itens Limpos</div>
            <div class="kpi-value" id="kpi-total">848</div>
            <div class="kpi-subtext">
                <span style="color: var(--split-treino)">Treino: 679 (80.1%)</span>
                <span style="color: var(--split-teste)">Teste: 169 (19.9%)</span>
            </div>
        </div>

        <div class="kpi-card">
            <div class="kpi-title">Legislação</div>
            <div class="kpi-value" id="kpi-legislação" style="color: var(--domain-legislação)">246</div>
            <div class="kpi-subtext">CF88, Código Penal, LGPD</div>
        </div>

        <div class="kpi-card">
            <div class="kpi-title">Saúde</div>
            <div class="kpi-value" id="kpi-saúde" style="color: var(--domain-saúde)">217</div>
            <div class="kpi-subtext">PCDTs & Portarias SUS</div>
        </div>

        <div class="kpi-card">
            <div class="kpi-title">Educação</div>
            <div class="kpi-value" id="kpi-educação" style="color: var(--domain-educação)">217</div>
            <div class="kpi-subtext">LDB, PNE, Editais ENEM</div>
        </div>

        <div class="kpi-card">
            <div class="kpi-title">Segurança Pública</div>
            <div class="kpi-value" id="kpi-segurança" style="color: var(--domain-segurança)">168</div>
            <div class="kpi-subtext">SUSP, Atlas Violência 2026</div>
        </div>
    </div>

    <!-- Analytics Charts -->
    <div class="analytics-container">
        <div class="analytics-title">
            <span>📊 Distribuição de Itens por Estrato e Tipo</span>
            <button class="btn" style="font-size: 11px; padding: 4px 8px;" onclick="toggleAnalytics()">Minimizar Painel</button>
        </div>
        <div class="bar-chart-grid" id="analytics-body">
            <div class="chart-box">
                <div class="chart-header">Distribuição por Domínio</div>
                <div id="chart-domain"></div>
            </div>
            <div class="chart-box">
                <div class="chart-header">Distribuição por Dificuldade</div>
                <div id="chart-difficulty"></div>
            </div>
            <div class="chart-box">
                <div class="chart-header">Divisão Treino / Teste</div>
                <div id="chart-split"></div>
            </div>
        </div>
    </div>

    <!-- Filters & Search -->
    <div class="filters-card">
        <div class="search-row">
            <div class="search-input-wrapper">
                <input type="text" id="search-input" class="search-input" placeholder="🔍 Buscar por palavra-chave em Pergunta, Resposta, Chunk ou Fontes..." oninput="onSearchChange()">
            </div>
            <select id="sort-select" class="select-input" onchange="onSortChange()">
                <option value="id-asc">Ordenar por: ID (Crescente)</option>
                <option value="id-desc">Ordenar por: ID (Decrescente)</option>
                <option value="qlen-desc">Ordenar por: Pergunta Mais Longa</option>
                <option value="alen-desc">Ordenar por: Resposta Mais Longa</option>
            </select>
            <select id="source-select" class="select-input" onchange="onSourceChange()">
                <option value="all">Todas as Fontes Documentais</option>
            </select>
        </div>

        <div class="filter-groups">
            <!-- Domínio -->
            <div class="filter-group">
                <span class="filter-group-title">Domínio:</span>
                <button class="pill-btn active" data-filter="domain" data-val="all" onclick="setFilter('domain', 'all')">Todos</button>
                <button class="pill-btn" data-filter="domain" data-val="legislacao" onclick="setFilter('domain', 'legislacao')">Legislação</button>
                <button class="pill-btn" data-filter="domain" data-val="saude" onclick="setFilter('domain', 'saude')">Saúde</button>
                <button class="pill-btn" data-filter="domain" data-val="edu" onclick="setFilter('domain', 'edu')">Educação</button>
                <button class="pill-btn" data-filter="domain" data-val="seguranca" onclick="setFilter('domain', 'seguranca')">Segurança</button>
            </div>

            <!-- Dificuldade -->
            <div class="filter-group">
                <span class="filter-group-title">Dificuldade:</span>
                <button class="pill-btn active" data-filter="diff" data-val="all" onclick="setFilter('diff', 'all')">Todas</button>
                <button class="pill-btn" data-filter="diff" data-val="factual" onclick="setFilter('diff', 'factual')">Factual</button>
                <button class="pill-btn" data-filter="diff" data-val="conceitual" onclick="setFilter('diff', 'conceitual')">Conceitual</button>
                <button class="pill-btn" data-filter="diff" data-val="aplicado" onclick="setFilter('diff', 'aplicado')">Aplicado</button>
            </div>

            <!-- Split -->
            <div class="filter-group">
                <span class="filter-group-title">Split:</span>
                <button class="pill-btn active" data-filter="split" data-val="all" onclick="setFilter('split', 'all')">Todos</button>
                <button class="pill-btn" data-filter="split" data-val="treino" onclick="setFilter('split', 'treino')">Treino (80%)</button>
                <button class="pill-btn" data-filter="split" data-val="teste" onclick="setFilter('split', 'teste')">Teste (20%)</button>
            </div>

            <!-- Group Type -->
            <div class="filter-group">
                <span class="filter-group-title">Sintese:</span>
                <button class="pill-btn active" data-filter="group" data-val="all" onclick="setFilter('group', 'all')">Todos</button>
                <button class="pill-btn" data-filter="group" data-val="single" onclick="setFilter('group', 'single')">Single-Chunk</button>
                <button class="pill-btn" data-filter="group" data-val="multi" onclick="setFilter('group', 'multi')">Multi-Chunk</button>
            </div>
        </div>
    </div>

    <!-- Results Meta -->
    <div class="results-meta">
        <span id="results-count">Exibindo 848 de 848 itens</span>
        <span>Página <span id="current-page-num">1</span> de <span id="total-pages-num">1</span></span>
    </div>

    <!-- Items Grid Container -->
    <div id="items-container" class="items-list"></div>

    <!-- Pagination Controls -->
    <div id="pagination-controls" class="pagination"></div>
</div>

<!-- Modal for Raw JSON View -->
<div id="json-modal" class="modal-overlay" onclick="closeModal(event)">
    <div class="modal-card" onclick="event.stopPropagation()">
        <div class="modal-header">
            <h3 style="font-size: 15px; font-weight: 700;">Payload JSON do Item</h3>
            <button class="btn" onclick="closeModal()">✕ Fechar</button>
        </div>
        <div class="modal-body" id="modal-json-content"></div>
    </div>
</div>

<script>
    // Embedded Dataset Payload
    const rawDataset = {json_data_str};
    
    // Application State
    let filteredItems = [...rawDataset];
    let currentPage = 1;
    const itemsPerPage = 20;

    let currentFilters = {{
        search: '',
        domain: 'all',
        diff: 'all',
        split: 'all',
        group: 'all',
        source: 'all',
        sort: 'id-asc'
    }};

    // DOM Elements
    const container = document.getElementById('items-container');
    const searchInput = document.getElementById('search-input');
    const sourceSelect = document.getElementById('source-select');
    const resultsCount = document.getElementById('results-count');
    const paginationControls = document.getElementById('pagination-controls');

    // Init
    window.addEventListener('DOMContentLoaded', () => {{
        populateSources();
        renderAnalytics();
        applyFilters();
    }});

    function populateSources() {{
        const sources = new Set();
        rawDataset.forEach(item => {{
            (item.fontes || []).forEach(f => sources.add(f));
        }});
        Array.from(sources).sort().forEach(src => {{
            const opt = document.createElement('option');
            opt.value = src;
            opt.textContent = src;
            sourceSelect.appendChild(opt);
        }});
    }}

    function renderAnalytics() {{
        // Domains
        const domainCounts = {{ legislacao: 0, saude: 0, edu: 0, seguranca: 0 }};
        const diffCounts = {{ factual: 0, conceitual: 0, aplicado: 0 }};
        const splitCounts = {{ treino: 0, teste: 0 }};

        rawDataset.forEach(item => {{
            if (domainCounts[item.dominio] !== undefined) domainCounts[item.dominio]++;
            if (diffCounts[item.nivel_dificuldade] !== undefined) diffCounts[item.nivel_dificuldade]++;
            if (splitCounts[item.split] !== undefined) splitCounts[item.split]++;
        }});

        const total = rawDataset.length || 1;

        // Render Domain Bar Chart
        const domainLabels = {{ legislacao: 'Legislação', saude: 'Saúde', edu: 'Educação', seguranca: 'Segurança' }};
        const domainColors = {{ legislacao: 'var(--domain-legislação)', saude: 'var(--domain-saúde)', edu: 'var(--domain-educação)', seguranca: 'var(--domain-segurança)' }};
        
        let domHtml = '';
        for (const [k, v] of Object.entries(domainCounts)) {{
            const pct = ((v / total) * 100).toFixed(1);
            domHtml += `
                <div class="bar-row">
                    <div class="bar-label"><span>${{domainLabels[k] || k}}</span><span>${{v}} (${{pct}}%)</span></div>
                    <div class="bar-track"><div class="bar-fill" style="width: ${{pct}}%; background-color: ${{domainColors[k]}}"></div></div>
                </div>
            `;
        }}
        document.getElementById('chart-domain').innerHTML = domHtml;

        // Render Difficulty Bar Chart
        const diffColors = {{ factual: 'var(--diff-factual)', conceitual: 'var(--diff-conceitual)', aplicado: 'var(--diff-aplicado)' }};
        let diffHtml = '';
        for (const [k, v] of Object.entries(diffCounts)) {{
            const pct = ((v / total) * 100).toFixed(1);
            diffHtml += `
                <div class="bar-row">
                    <div class="bar-label"><span style="text-transform: capitalize">${{k}}</span><span>${{v}} (${{pct}}%)</span></div>
                    <div class="bar-track"><div class="bar-fill" style="width: ${{pct}}%; background-color: ${{diffColors[k]}}"></div></div>
                </div>
            `;
        }}
        document.getElementById('chart-difficulty').innerHTML = diffHtml;

        // Render Split Bar Chart
        let splitHtml = `
            <div class="bar-row">
                <div class="bar-label"><span>Treino (80%)</span><span>${{splitCounts.treino}} (${{((splitCounts.treino/total)*100).toFixed(1)}}%)</span></div>
                <div class="bar-track"><div class="bar-fill" style="width: ${{((splitCounts.treino/total)*100).toFixed(1)}}%; background-color: var(--split-treino)"></div></div>
            </div>
            <div class="bar-row">
                <div class="bar-label"><span>Teste (20%)</span><span>${{splitCounts.teste}} (${{((splitCounts.teste/total)*100).toFixed(1)}}%)</span></div>
                <div class="bar-track"><div class="bar-fill" style="width: ${{((splitCounts.teste/total)*100).toFixed(1)}}%; background-color: var(--split-teste)"></div></div>
            </div>
        `;
        document.getElementById('chart-split').innerHTML = splitHtml;
    }}

    function setFilter(type, val) {{
        currentFilters[type] = val;
        
        // Update pill UI
        document.querySelectorAll(`.pill-btn[data-filter="${{type}}"]`).forEach(btn => {{
            if (btn.getAttribute('data-val') === val) {{
                btn.classList.add('active');
            }} else {{
                btn.classList.remove('active');
            }}
        }});

        applyFilters();
    }}

    function onSearchChange() {{
        currentFilters.search = searchInput.value.trim().toLowerCase();
        applyFilters();
    }}

    function onSortChange() {{
        currentFilters.sort = document.getElementById('sort-select').value;
        applyFilters();
    }}

    function onSourceChange() {{
        currentFilters.source = sourceSelect.value;
        applyFilters();
    }}

    function resetFilters() {{
        currentFilters = {{
            search: '',
            domain: 'all',
            diff: 'all',
            split: 'all',
            group: 'all',
            source: 'all',
            sort: 'id-asc'
        }};
        searchInput.value = '';
        sourceSelect.value = 'all';
        document.getElementById('sort-select').value = 'id-asc';

        document.querySelectorAll('.pill-btn').forEach(btn => {{
            if (btn.getAttribute('data-val') === 'all') {{
                btn.classList.add('active');
            }} else {{
                btn.classList.remove('active');
            }}
        }});

        applyFilters();
    }}

    function applyFilters() {{
        filteredItems = rawDataset.filter(item => {{
            // Domain
            if (currentFilters.domain !== 'all' && item.dominio !== currentFilters.domain) return false;
            // Difficulty
            if (currentFilters.diff !== 'all' && item.nivel_dificuldade !== currentFilters.diff) return false;
            // Split
            if (currentFilters.split !== 'all' && item.split !== currentFilters.split) return false;
            // Group Type
            if (currentFilters.group !== 'all' && item.group_type !== currentFilters.group) return false;
            // Source
            if (currentFilters.source !== 'all' && !(item.fontes || []).includes(currentFilters.source)) return false;

            // Search
            if (currentFilters.search) {{
                const q = currentFilters.search;
                const matchId = item.id.toLowerCase().includes(q);
                const matchQ = item.pergunta.toLowerCase().includes(q);
                const matchA = item.resposta_referencia.toLowerCase().includes(q);
                const matchChunk = (item.chunk_texto || []).some(c => c.toLowerCase().includes(q));
                const matchSource = (item.fontes || []).some(s => s.toLowerCase().includes(q));
                if (!matchId && !matchQ && !matchA && !matchChunk && !matchSource) return false;
            }}

            return true;
        }});

        // Sort
        filteredItems.sort((a, b) => {{
            if (currentFilters.sort === 'id-asc') return a.id.localeCompare(b.id);
            if (currentFilters.sort === 'id-desc') return b.id.localeCompare(a.id);
            if (currentFilters.sort === 'qlen-desc') return b.pergunta.length - a.pergunta.length;
            if (currentFilters.sort === 'alen-desc') return b.resposta_referencia.length - a.resposta_referencia.length;
            return 0;
        }});

        currentPage = 1;
        renderItems();
    }}

    function renderItems() {{
        const total = filteredItems.length;
        const totalPages = Math.ceil(total / itemsPerPage) || 1;

        resultsCount.textContent = `Exibindo ${{filteredItems.length}} de ${{rawDataset.length}} itens`;
        document.getElementById('current-page-num').textContent = currentPage;
        document.getElementById('total-pages-num').textContent = totalPages;

        if (total === 0) {{
            container.innerHTML = `
                <div style="text-align: center; padding: 60px 20px; background: var(--bg-card); border: 1px solid var(--border-color); border-radius: var(--radius-md); color: var(--text-muted);">
                    <div style="font-size: 32px; margin-bottom: 12px;">🔍</div>
                    <div style="font-size: 16px; font-weight: 700; margin-bottom: 6px;">Nenhum item encontrado</div>
                    <div style="font-size: 13px;">Tente ajustar seus filtros de busca ou clique em 'Resetar Filtros'.</div>
                </div>
            `;
            paginationControls.innerHTML = '';
            return;
        }}

        const startIdx = (currentPage - 1) * itemsPerPage;
        const pageItems = filteredItems.slice(startIdx, startIdx + itemsPerPage);

        let html = '';
        pageItems.forEach(item => {{
            const domainClass = `badge-domain-${{item.dominio}}`;
            const diffClass = `badge-diff-${{item.nivel_dificuldade}}`;
            const splitClass = `badge-split-${{item.split}}`;

            const domainLabel = {{ legislacao: 'Legislação', saude: 'Saúde', edu: 'Educação', seguranca: 'Segurança' }}[item.dominio] || item.dominio;

            const fuentesChips = (item.fontes || []).map(f => `<span class="source-chip">📄 ${{f}}</span>`).join(' ');
            const chunkCount = (item.chunk_ids || []).length;

            const contextText = (item.chunk_texto || []).map((c, i) => `[Chunk ${{i+1}}]: ${{c}}`).join('\\n\\n');

            html += `
                <div class="item-card">
                    <div class="item-header">
                        <span class="item-id">${{item.id}}</span>
                        <div class="badge-list">
                            <span class="badge ${{domainClass}}">${{domainLabel}}</span>
                            <span class="badge ${{diffClass}}">${{item.nivel_dificuldade}}</span>
                            <span class="badge ${{splitClass}}">${{item.split}}</span>
                            <span class="badge badge-group">${{item.group_type || 'single'}} (${{chunkCount}} chunk${{chunkCount > 1 ? 's' : ''}})</span>
                        </div>
                    </div>

                    <div class="item-section">
                        <div class="item-label">Pergunta (Prompt)</div>
                        <div class="question-text">${{highlightSearch(item.pergunta)}}</div>
                    </div>

                    <div class="item-section">
                        <div class="item-label">Resposta de Referência (Ground Truth)</div>
                        <div class="answer-box">${{highlightSearch(item.resposta_referencia)}}</div>
                    </div>

                    <div class="item-section">
                        <div class="item-label">Fontes Documentais</div>
                        <div class="sources-chips">${{fuentesChips}}</div>
                    </div>

                    <details>
                        <summary>
                            <span>📖 Ver Contexto dos Chunks (${{chunkCount}} bloco${{chunkCount > 1 ? 's' : ''}})</span>
                            <span style="font-size: 11px; color: var(--text-dim);">Clique para expandir</span>
                        </summary>
                        <div class="context-content">${{highlightSearch(contextText)}}</div>
                    </details>

                    <div style="margin-top: 12px; display: flex; justify-content: flex-end; gap: 8px;">
                        <button class="btn" style="font-size: 11px; padding: 4px 10px;" onclick="copyItemQA('${{item.id}}')">📋 Copiar Par Q&A</button>
                        <button class="btn" style="font-size: 11px; padding: 4px 10px;" onclick="openModal('${{item.id}}')">🔍 Ver JSON Completo</button>
                    </div>
                </div>
            `;
        }});

        container.innerHTML = html;
        renderPagination(totalPages);
    }}

    function highlightSearch(text) {{
        if (!currentFilters.search || !text) return text;
        const q = currentFilters.search.replace(/[.*+?^${{}}()|[\\]\\\\]/g, '\\\\$&');
        const regex = new RegExp(`(${{q}})`, 'gi');
        return text.replace(regex, '<span class="highlight">$1</span>');
    }}

    function renderPagination(totalPages) {{
        if (totalPages <= 1) {{
            paginationControls.innerHTML = '';
            return;
        }}

        let html = '';
        html += `<button class="page-btn" ${{currentPage === 1 ? 'disabled' : ''}} onclick="goToPage(1)">«</button>`;
        html += `<button class="page-btn" ${{currentPage === 1 ? 'disabled' : ''}} onclick="goToPage(${{currentPage - 1}})">‹</button>`;

        let start = Math.max(1, currentPage - 2);
        let end = Math.min(totalPages, currentPage + 2);

        for (let p = start; p <= end; p++) {{
            html += `<button class="page-btn ${{p === currentPage ? 'active' : ''}}" onclick="goToPage(${{p}})">${{p}}</button>`;
        }}

        html += `<button class="page-btn" ${{currentPage === totalPages ? 'disabled' : ''}} onclick="goToPage(${{currentPage + 1}})">›</button>`;
        html += `<button class="page-btn" ${{currentPage === totalPages ? 'disabled' : ''}} onclick="goToPage(${{totalPages}})">»</button>`;

        paginationControls.innerHTML = html;
    }}

    function goToPage(page) {{
        currentPage = page;
        renderItems();
        window.scrollTo({{ top: 400, behavior: 'smooth' }});
    }}

    function toggleTheme() {{
        const html = document.documentElement;
        const current = html.getAttribute('data-theme');
        const next = current === 'dark' ? 'light' : 'dark';
        html.setAttribute('data-theme', next);
    }}

    function toggleAnalytics() {{
        const body = document.getElementById('analytics-body');
        if (body.style.display === 'none') {{
            body.style.display = 'grid';
        }} else {{
            body.style.display = 'none';
        }}
    }}

    function openModal(id) {{
        const item = rawDataset.find(i => i.id === id);
        if (item) {{
            document.getElementById('modal-json-content').textContent = JSON.stringify(item, null, 2);
            document.getElementById('json-modal').style.display = 'flex';
        }}
    }}

    function closeModal() {{
        document.getElementById('json-modal').style.display = 'none';
    }}

    function copyItemQA(id) {{
        const item = rawDataset.find(i => i.id === id);
        if (item) {{
            const text = `ID: ${{item.id}}\\nDomínio: ${{item.dominio}} | Dificuldade: ${{item.nivel_dificuldade}} | Split: ${{item.split}}\\n\\nPergunta:\\n${{item.pergunta}}\\n\\nResposta de Referência:\\n${{item.resposta_referencia}}`;
            navigator.clipboard.writeText(text).then(() => {{
                alert(`Par Q&A do item [${{id}}] copiado para a área de transferência!`);
            }});
        }}
    }}

    function exportFilteredJSON() {{
        const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(filteredItems, null, 2));
        const downloadAnchor = document.createElement('a');
        downloadAnchor.setAttribute("href", dataStr);
        downloadAnchor.setAttribute("download", `govbench_br_filtered_${{filteredItems.length}}_items.json`);
        document.body.appendChild(downloadAnchor);
        downloadAnchor.click();
        downloadAnchor.remove();
    }}
</script>
</body>
</html>
"""

    with open(out_html, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"Visualizador interativo gerado com sucesso!")
    print(f"Arquivo salvo em: {out_html}")

if __name__ == "__main__":
    build_visualizer()
