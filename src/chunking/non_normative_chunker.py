import re
import os

# Header patterns
SECTION_HEADER_PATTERN = re.compile(
    r'^(?:#{1,6}\s+|(?:\d+\.|\d+\)\s+)[A-Z\xc0-\xff]{2,}|Quadro\s+\d+|Tabela\s+\d+)',
    re.IGNORECASE
)

PAGE_MARKER_PATTERN = re.compile(r'^##\s+Página\s+\d+', re.IGNORECASE)

def chunk_non_normative_document(
    file_path: str,
    domain: str,
    max_chars: int = 1800,
    min_chars: int = 300
) -> list[dict]:
    """
    Segmenta um documento não-normativo (PCDT, Atlas da Violência) por subseções temáticas.
    Preserva tabelas Markdown e quadros de forma íntegra.
    Agrupa parágrafos dentro de limites de tamanho (min_chars a max_chars).
    """
    filename = os.path.basename(file_path)
    doc_id = os.path.splitext(filename)[0]
    
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    lines = text.splitlines()
    
    chunks = []
    current_section = "Introdução / Geral"
    current_paragraphs = []
    current_char_count = 0
    chunk_counter = 0

    def emit_chunk(content: str, is_table: bool = False, section_name: str = None):
        nonlocal chunk_counter
        clean_content = content.strip()
        if not clean_content:
            return
            
        chunk_counter += 1
        chunks.append({
            "chunk_id": f"{domain}_{doc_id}_chunk_{chunk_counter:04d}",
            "domain": domain,
            "source_document": filename,
            "chunk_index": chunk_counter,
            "chunk_type": "table_chunk" if is_table else "thematic_subsection",
            "article_ref": None,
            "hierarchy": [section_name or current_section],
            "section_title": section_name or current_section,
            "content": clean_content,
            "char_count": len(clean_content),
            "word_count": len(clean_content.split()),
            "has_table": is_table or ("|" in clean_content and "-|-" in clean_content)
        })

    # Bloco para agrupar tabelas
    in_table = False
    table_lines = []

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.strip()

        # Detecta início/continuação de Tabela Markdown
        if stripped.startswith("|") and ("|" in stripped[1:]):
            in_table = True
            table_lines.append(stripped)
            i += 1
            continue
        elif in_table:
            # Fim de bloco de tabela
            in_table = False
            # Emitir parágrafos acumulados antes da tabela
            if current_paragraphs:
                accumulated = "\n\n".join(current_paragraphs)
                emit_chunk(accumulated)
                current_paragraphs = []
                current_char_count = 0
            
            # Emitir tabela como chunk de tabela
            table_content = "\n".join(table_lines)
            emit_chunk(table_content, is_table=True)
            table_lines = []

        if not stripped:
            i += 1
            continue

        # Detecta marcador de página (ex: ## Página 10) - atualiza contexto se não houver seção melhor
        if PAGE_MARKER_PATTERN.match(stripped):
            # Não forçamos novo chunk apenas pela página, mas registramos
            i += 1
            continue

        # Detecta novo cabeçalho de seção (ex: # 1. DIAGNÓSTICO, ## Transtornos, etc.)
        if SECTION_HEADER_PATTERN.match(stripped):
            # Emitir o que já temos acumulado se ultrapassar min_chars
            if current_paragraphs:
                accumulated = "\n\n".join(current_paragraphs)
                emit_chunk(accumulated)
                current_paragraphs = []
                current_char_count = 0

            # Atualiza o nome da seção ativa
            clean_heading = re.sub(r'^#+\s*', '', stripped).strip()
            current_section = clean_heading
            i += 1
            continue

        # Acumula linha como parte do parágrafo/texto
        current_paragraphs.append(stripped)
        current_char_count += len(stripped) + 2

        # Se atingiu o tamanho máximo estipulado, emite o chunk
        if current_char_count >= max_chars:
            accumulated = "\n\n".join(current_paragraphs)
            emit_chunk(accumulated)
            current_paragraphs = []
            current_char_count = 0

        i += 1

    # Emite remanescente após o loop
    if in_table and table_lines:
        if current_paragraphs:
            accumulated = "\n\n".join(current_paragraphs)
            emit_chunk(accumulated)
            current_paragraphs = []
        table_content = "\n".join(table_lines)
        emit_chunk(table_content, is_table=True)

    if current_paragraphs:
        accumulated = "\n\n".join(current_paragraphs)
        emit_chunk(accumulated)

    return chunks
