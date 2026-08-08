"""
govbench_common.py
-------------------
Prompt, templates por nível de dificuldade e utilitários compartilhados
entre 02_test_model_generation.py (piloto/comparação de modelos) e
03_generate_govbench_br.py (geração oficial em escala).

Manter isso em um único lugar garante que o prompt validado no piloto é
EXATAMENTE o mesmo usado na geração em escala -- nunca editar o prompt em
só um dos dois scripts.
"""

import json
import re
import time

from litellm import completion
from dotenv import load_dotenv

load_dotenv()  # Carrega .env automaticamente (GEMINI_API_KEY, etc.)

# --------------------------------------------------------------------------
# PROMPT GERAÇÃO
# --------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "Você é um especialista em elaborar itens de avaliação de alto nível (perguntas e "
    "respostas de referência) sobre documentos e legislações oficiais do Brasil, para "
    "compor o benchmark GovBench-BR.\n\n"
    "DIRETRIZES DE QUALIDADE:\n"
    "1. Rigor Textual: A pergunta e a resposta devem ser estritamente fundamentadas nos trechos fornecidos.\n"
    "2. Pergunta Autocontida: Evite expressões vagas como 'segundo o trecho'. Especifique a norma ou o tema na pergunta (ex: 'De acordo com o Art. 163 da Constituição Federal...').\n"
    "3. Resposta Completa: A resposta de referência deve ser clara, técnica, precisa e abranger todos os pontos exigidos na pergunta.\n"
    "4. Formato Estrito: Sua resposta DEVE ser ÚNICA e EXCLUSIVAMENTE um objeto JSON válido.\n\n"
    "EXEMPLO DE SAÍDA ESPERADA (1-Shot):\n"
    "{\n"
    '  "pergunta": "De acordo com o Art. 163 da Constituição Federal, qual instrumento normativo deve dispor sobre a sustentabilidade da dívida pública e quais elementos ele deve conter?",\n'
    '  "resposta_referencia": "A sustentabilidade da dívida pública deve ser disposta por meio de lei complementar. Essa lei deve especificar os indicadores de apuração, a compatibilidade dos resultados fiscais com a trajetória da dívida e as medidas de ajuste necessárias.",\n'
    '  "trechos_usados": ["Art. 163, VIII"]\n'
    "}"
)

INSTRUCOES_POR_NIVEL = {
    "factual": (
        "Nível de dificuldade: FACTUAL.\n"
        "Elabore UMA pergunta cuja resposta esteja diretamente contida no "
        "trecho fornecido, exigindo apenas extração/recordação direta, sem "
        "necessidade de interpretação ou inferência."
    ),
    "conceitual": (
        "Nível de dificuldade: CONCEITUAL.\n"
        "Elabore UMA pergunta que exija explicar, definir ou comparar um "
        "conceito presente no(s) trecho(s), em vez de apenas repetir o "
        "texto literalmente. A resposta deve demandar paráfrase ou síntese "
        "de uma ideia, não cópia direta de uma frase."
    ),
    "aplicado": (
        "Nível de dificuldade: APLICADO.\n"
        "Elabore UMA pergunta em formato de cenário/caso hipotético cuja "
        "resposta correta exija combinar informações de TODOS os trechos "
        "fornecidos (não pode ser respondida usando só um deles). Não "
        "invente fatos fora dos trechos fornecidos."
    ),
}

OUTPUT_SCHEMA_INSTRUCTIONS = (
    "Responda apenas com um objeto JSON com exatamente estas chaves:\n"
    '{"pergunta": "...", "resposta_referencia": "...", '
    '"trechos_usados": ["<article_ref ou identificador do trecho>", ...]}'
)


def build_prompt(task: dict) -> str:
    chunks = task["chunks"]
    parts = [INSTRUCOES_POR_NIVEL[task["nivel_dificuldade"]], ""]
    for i, c in enumerate(chunks, start=1):
        label = c.get("article_ref") or c.get("section_title") or c["chunk_id"]
        parts.append(f"--- Trecho {i} (fonte: {c['source_document']} | {label}) ---")
        parts.append(c["content"])
        parts.append("")
    parts.append(OUTPUT_SCHEMA_INSTRUCTIONS)
    return "\n".join(parts)


def extract_json(text: str) -> dict:
    if not text or not text.strip():
        raise ValueError("Resposta vazia retornada pelo modelo.")
    text = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)
    else:
        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            text = text[first_brace:last_brace + 1]
    return json.loads(text)


 
# --------------------------------------------------------------------------
# LLM-AS-JUDGE (auditoria independente dos pares QA já gerados)
# --------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = (
    "Você é um auditor independente e rigoroso de um benchmark de perguntas e "
    "respostas sobre documentos oficiais brasileiros (GovBench-BR). Você NÃO "
    "gerou este item -- sua função é AVALIAR criticamente se ele está correto, "
    "sem usar seu próprio conhecimento de mundo para completar lacunas.\n\n"
    "CRITÉRIOS DE AVALIAÇÃO:\n"
    "1. fundamentado_no_trecho: a resposta de referência pode ser INTEIRAMENTE "
    "derivada do(s) trecho(s) fornecido(s), sem precisar de nenhum fato externo?\n"
    "2. informacao_extra_nao_presente: a resposta contém algum fato, número "
    "ou detalhe factual que NÃO aparece no(s) trecho(s) fornecido(s)? "
    "(Nota: A citação do nome oficial do documento/lei/norma fornecido no cabeçalho "
    "do trecho NÃO é considerada informação extra externa).\n"
    "3. nivel_dificuldade_adequado: a complexidade da pergunta é compatível "
    "com o nível declarado (factual = extração direta; conceitual = "
    "explicação/definição; aplicado = exige combinar TODOS os trechos "
    "fornecidos, não só um deles)?\n"
    "4. resposta_completa_precisa: a resposta cobre todos os pontos exigidos "
    "pela pergunta, sem ser vaga nem incompleta?\n\n"
    "Responda SEMPRE em português, apenas com um objeto JSON válido, sem "
    "texto antes ou depois, no formato:\n"
    "{\n"
    '  "fundamentado_no_trecho": true/false,\n'
    '  "informacao_extra_nao_presente": true/false,\n'
    '  "nivel_dificuldade_adequado": true/false,\n'
    '  "resposta_completa_precisa": true/false,\n'
    '  "veredito": "aprovado" | "revisar" | "rejeitar",\n'
    '  "justificativa": "1-2 frases, apontando o problema específico se houver"\n'
    "}\n"
    "Regra para o veredito: 'rejeitar' se informacao_extra_nao_presente=true ou "
    "fundamentado_no_trecho=false; 'revisar' se algum outro critério falhar; "
    "'aprovado' só se os 4 critérios passarem."
)


def build_judge_prompt(item: dict) -> str:
    fontes = item.get("fontes", [])
    fontes_str = f" | Fonte(s) Oficial(is): {', '.join(fontes)}" if fontes else ""
    parts = [
        f"Nível de dificuldade declarado: {item['nivel_dificuldade'].upper()}",
        "",
    ]
    chunk_textos = item.get("chunk_texto", [])
    if isinstance(chunk_textos, str):
        chunk_textos = [chunk_textos]
    for i, ct in enumerate(chunk_textos, start=1):
        parts.append(f"--- Trecho {i}{fontes_str} ---")
        parts.append(ct)
        parts.append("")
    parts.append(f"PERGUNTA: {item['pergunta']}")
    parts.append(f"RESPOSTA DE REFERÊNCIA: {item['resposta_referencia']}")
    return "\n".join(parts)
 
 
def call_judge(model: str, item: dict, max_retries: int = 2, timeout: int = 120) -> dict:
    """Chama um modelo-juiz para avaliar UM par QA já gerado. Independente de
    call_model (geração) -- não reaproveita retries para não arriscar
    regressão no que já está validado em 02/03."""
    prompt = build_judge_prompt(item)
    out = {
        "veredito": None,
        "fundamentado_no_trecho": None,
        "informacao_extra_nao_presente": None,
        "nivel_dificuldade_adequado": None,
        "resposta_completa_precisa": None,
        "justificativa": None,
        "raw_response": None,
        "error": None,
        "parse_ok": False,
    }
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            kwargs = {
                "model": model,
                "messages": [
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.0,
                "max_tokens": 512,
                "timeout": timeout,
            }
            try:
                kwargs["response_format"] = {"type": "json_object"}
            except Exception:
                pass

            resp = completion(**kwargs)
            msg = resp["choices"][0]["message"]
            content = getattr(msg, "content", None)
            if content is None and isinstance(msg, dict):
                content = msg.get("content")
            content = content or ""
            
            out["raw_response"] = content
            parsed = extract_json(content)
            out.update({k: parsed.get(k) for k in (
                "veredito", "fundamentado_no_trecho", "informacao_extra_nao_presente",
                "nivel_dificuldade_adequado", "resposta_completa_precisa", "justificativa",
            )})
            out["parse_ok"] = out["veredito"] in ("aprovado", "revisar", "rejeitar")
            return out
        except Exception as e:
            last_err = str(e)
            time.sleep(1.5 * (attempt + 1))
    out["error"] = last_err
    return out


def call_model(model: str, task: dict, max_retries: int = 2, timeout: int = 120) -> dict:
    """Chama o modelo para UMA tarefa e devolve um dict com o resultado bruto
    (usado tanto pelo piloto quanto pela geração oficial)."""
    prompt = build_prompt(task)
    out = {
        "parse_ok": False,
        "pergunta": None,
        "resposta_referencia": None,
        "trechos_usados": None,
        "raw_response": None,
        "error": None,
        "latency_s": None,
    }

    last_err = None
    for attempt in range(max_retries + 1):
        t0 = time.time()
        try:
            kwargs = {
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.4,
                "max_tokens": 2048,
                "timeout": timeout,
            }
            try:
                kwargs["response_format"] = {"type": "json_object"}
            except Exception:
                pass

            resp = completion(**kwargs)
            latency = time.time() - t0
            msg = resp["choices"][0]["message"]
            content = getattr(msg, "content", None)
            if content is None and isinstance(msg, dict):
                content = msg.get("content")
            content = content or ""
            out["raw_response"] = content
            out["latency_s"] = round(latency, 2)

            parsed = extract_json(content)
            out["pergunta"] = parsed.get("pergunta")
            out["resposta_referencia"] = parsed.get("resposta_referencia")
            out["trechos_usados"] = parsed.get("trechos_usados")
            out["parse_ok"] = bool(out["pergunta"] and out["resposta_referencia"])
            return out
        except Exception as e:
            last_err = str(e)
            time.sleep(1.5 * (attempt + 1))

    out["error"] = last_err
    return out
