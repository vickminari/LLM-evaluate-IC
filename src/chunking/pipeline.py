import os
import glob
import json
from tqdm import tqdm

from .normative_chunker import chunk_normative_document
from .non_normative_chunker import chunk_non_normative_document

# Mapeamento de documentos normativos explícitos
NORMATIVE_FILE_NAMES = {
    "constituicao_federal.md",
    "del2848.md",
    "lgpd.md",
    "lei9394-1996.md",
    "lei13005--2014.md",
    "edital_enem.md",
    "susp.md"
}

def is_normative(file_path: str) -> bool:
    filename = os.path.basename(file_path)
    return filename in NORMATIVE_FILE_NAMES

def run_pipeline(docs_base_dir: str, output_dir: str):
    """
    Executa a pipeline completa de chunkerização sobre os 4 domínios da IC:
    Legislação, Educação, Saúde e Segurança.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    domain_chunks = {
        "legislacao": [],
        "edu": [],
        "saude": [],
        "seguranca": []
    }
    
    file_stats = []
    
    # 1. Varre os diretórios de cada domínio
    domains = ["legislacao", "edu", "saude", "seguranca"]
    
    for domain in domains:
        domain_dir = os.path.join(docs_base_dir, domain)
        if not os.path.exists(domain_dir):
            print(f"[Aviso] Diretório não encontrado: {domain_dir}")
            continue
            
        # Coleta arquivos .md
        md_files = []
        if domain == "saude":
            # Inclui arquivos em saude/ e saude/pcds_md/
            md_files.extend(glob.glob(os.path.join(domain_dir, "*.md")))
            md_files.extend(glob.glob(os.path.join(domain_dir, "pcds_md", "*.md")))
        else:
            md_files.extend(glob.glob(os.path.join(domain_dir, "*.md")))
            
        print(f"\n---> Processando Domínio '{domain}': {len(md_files)} arquivo(s) encontrado(s)")
        
        for file_path in tqdm(md_files, desc=f"Chunking {domain}"):
            rel_path = os.path.relpath(file_path, docs_base_dir)
            normative = is_normative(file_path)
            
            try:
                if normative:
                    chunks = chunk_normative_document(file_path, domain)
                else:
                    chunks = chunk_non_normative_document(file_path, domain)
                    
                domain_chunks[domain].extend(chunks)
                
                # Registra estatísticas
                char_counts = [c["char_count"] for c in chunks]
                word_counts = [c["word_count"] for c in chunks]
                
                file_stats.append({
                    "domain": domain,
                    "file": os.path.basename(file_path),
                    "relative_path": rel_path,
                    "type": "normative" if normative else "non_normative",
                    "total_chunks": len(chunks),
                    "avg_char_count": round(sum(char_counts) / len(char_counts), 2) if chunks else 0,
                    "avg_word_count": round(sum(word_counts) / len(word_counts), 2) if chunks else 0
                })
            except Exception as e:
                print(f"[Erro] Falha ao processar {file_path}: {e}")

    # 2. Salva saídas JSONL por domínio e arquivo consolidado
    all_chunks = []
    summary_by_domain = {}
    
    for domain, chunks in domain_chunks.items():
        domain_file = os.path.join(output_dir, f"chunks_{domain}.jsonl")
        with open(domain_file, "w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
                
        all_chunks.extend(chunks)
        
        char_counts = [c["char_count"] for c in chunks]
        word_counts = [c["word_count"] for c in chunks]
        
        summary_by_domain[domain] = {
            "total_chunks": len(chunks),
            "avg_char_count": round(sum(char_counts) / len(char_counts), 2) if chunks else 0,
            "avg_word_count": round(sum(word_counts) / len(word_counts), 2) if chunks else 0
        }
        print(f"Salvo: {domain_file} ({len(chunks)} chunks)")

    # 3. Salva all_chunks.jsonl
    all_file = os.path.join(output_dir, "all_chunks.jsonl")
    with open(all_file, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"\nSalvo dataset consolidado: {all_file} (Total: {len(all_chunks)} chunks)")

    # 4. Salva relatório estatístico
    report_file = os.path.join(output_dir, "chunking_report.json")
    report_data = {
        "total_chunks_all_domains": len(all_chunks),
        "summary_by_domain": summary_by_domain,
        "files_detail": file_stats
    }
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)
    print(f"Salvo relatório de chunkerização: {report_file}")

if __name__ == "__main__":
    workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    docs_dir = os.path.join(workspace_root, "docs")
    output_dir = os.path.join(workspace_root, "data", "chunks")
    
    print(f"Iniciando pipeline de chunkerização em: {docs_dir}")
    print(f"Diretório de saída: {output_dir}")
    
    run_pipeline(docs_dir, output_dir)
