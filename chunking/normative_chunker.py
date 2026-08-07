import re
import os

# Regex para identificar artigos na legislação brasileira
# Ex: Art. 1º, Art. 1º-A, Art. 121, Artigo 5º, Art. 2.º, Art. 144-B
ART_PATTERN = re.compile(
    r'^\s*(?:Art\.|Artigo)\s*(\d+[A-Z0-9\.\-ºª°]*)\s*[-–—\.]?\s*',
    re.IGNORECASE
)

# Regex para cláusulas/itens numerados em editais (ex: 1.1, 1.2, 1.5.1)
ITEM_PATTERN = re.compile(
    r'^\s*(\d+\.\d+(?:\.\d+)?)\s+([A-Z\xc0-\xff].*)'
)

# Regex para capturar níveis hierárquicos (TÍTULO, CAPÍTULO, SEÇÃO, SUBSEÇÃO, PARTE)
HIERARCHY_PATTERN = re.compile(
    r'^\s*(PARTE\s+[A-Z]+|TÍTULO\s+[IVXLCDM]+|CAPÍTULO\s+[IVXLCDM]+|SEÇÃO\s+[IVXLCDM]+|SUBSEÇÃO\s+[IVXLCDM]+|LIVRO\s+[IVXLCDM]+)\b',
    re.IGNORECASE
)

def chunk_normative_document(file_path: str, domain: str) -> list[dict]:
    """
    Segmenta um documento normativo em chunks por artigo (ou item normativo).
    Preserva parágrafos (§), incisos (I -) e alíneas (a)) no mesmo chunk do artigo.
    Rastreia a hierarquia funcional (Parte, Título, Capítulo, Seção).
    """
    filename = os.path.basename(file_path)
    doc_id = os.path.splitext(filename)[0]
    
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
        
    chunks = []
    current_hierarchy = []
    
    current_chunk_lines = []
    current_art_ref = None
    current_section_title = ""
    pending_subtitle = ""
    chunk_counter = 0
    
    # Verifica se o documento usa estilo Art. ou estilo Item (ex: 1.1 em editais)
    has_art_style = any(ART_PATTERN.search(line) for line in lines)
    
    for line in lines:
        stripped = line.strip()
        
        if not stripped:
            if current_chunk_lines and current_chunk_lines[-1] != "":
                current_chunk_lines.append("")
            continue
            
        # 1. Verifica se a linha é um título de hierarquia (ex: TÍTULO I, CAPÍTULO II)
        h_match = HIERARCHY_PATTERN.search(stripped)
        if h_match:
            h_type = h_match.group(1).upper()
            if "PARTE" in h_type or "LIVRO" in h_type:
                current_hierarchy = [stripped]
            elif "TÍTULO" in h_type:
                current_hierarchy = [h for h in current_hierarchy if "PARTE" in h or "LIVRO" in h] + [stripped]
            elif "CAPÍTULO" in h_type:
                current_hierarchy = [h for h in current_hierarchy if "PARTE" in h or "LIVRO" in h or "TÍTULO" in h] + [stripped]
            elif "SEÇÃO" in h_type:
                current_hierarchy = [h for h in current_hierarchy if "CAPÍTULO" in h or "TÍTULO" in h or "PARTE" in h] + [stripped]
            continue

        # 2. Testa correspondência de início de Artigo (ou Item)
        art_match = ART_PATTERN.search(stripped) if has_art_style else ITEM_PATTERN.search(stripped)
        
        if art_match:
            # Finaliza o chunk anterior, se houver
            if current_chunk_lines:
                chunk_content = "\n".join(current_chunk_lines).strip()
                if chunk_content:
                    chunk_counter += 1
                    chunks.append({
                        "chunk_id": f"{domain}_{doc_id}_chunk_{chunk_counter:04d}",
                        "domain": domain,
                        "source_document": filename,
                        "chunk_index": chunk_counter,
                        "chunk_type": "normative_article" if has_art_style else "normative_item",
                        "article_ref": current_art_ref or f"Chunk {chunk_counter}",
                        "hierarchy": list(current_hierarchy),
                        "section_title": current_section_title,
                        "content": chunk_content,
                        "char_count": len(chunk_content),
                        "word_count": len(chunk_content.split()),
                        "has_table": "|" in chunk_content and "-|-" in chunk_content
                    })
            
            # Inicia novo chunk
            current_art_ref = art_match.group(0).strip()
            current_section_title = pending_subtitle.strip()
            pending_subtitle = ""
            current_chunk_lines = [stripped]
        else:
            # Se ainda não encontramos nenhum artigo no documento, acumulamos como subtítulo/preâmbulo
            if current_art_ref is None:
                if len(stripped) < 100 and not stripped.startswith("§") and not stripped.startswith("(") and not stripped.startswith("I "):
                    pending_subtitle = stripped
                continue
            
            # Se for linha subsequente dentro do artigo (parágrafos, incisos, alíneas, texto)
            current_chunk_lines.append(stripped)

    # Emite o último chunk
    if current_chunk_lines:
        chunk_content = "\n".join(current_chunk_lines).strip()
        if chunk_content:
            chunk_counter += 1
            chunks.append({
                "chunk_id": f"{domain}_{doc_id}_chunk_{chunk_counter:04d}",
                "domain": domain,
                "source_document": filename,
                "chunk_index": chunk_counter,
                "chunk_type": "normative_article" if has_art_style else "normative_item",
                "article_ref": current_art_ref or f"Chunk {chunk_counter}",
                "hierarchy": list(current_hierarchy),
                "section_title": current_section_title,
                "content": chunk_content,
                "char_count": len(chunk_content),
                "word_count": len(chunk_content.split()),
                "has_table": "|" in chunk_content and "-|-" in chunk_content
            })
            
    return chunks
