#!/usr/bin/env python3
"""
09_evaluate_llms.py
---------------------
Inferência closed-book nos itens de teste do GovBench-BR. Gera as respostas de cada
modelo -- o cálculo de métricas (ROUGE-L, EM, LLM-as-Judge) 
fica para o próximo script.

DOIS MOTORES DE INFERÊNCIA:
  - unsloth: para os nomes reservados "qwen_base" e "qwen_finetuned" --
    carrega o Qwen3.5-9B (com ou sem o adaptador LoRA v2) diretamente neste
    processo. Só UM modelo Unsloth por execução: 8GB de VRAM não permite
    carregar dois modelos de 9B em sequência com segurança no mesmo
    processo -- rode qwen_base e qwen_finetuned em execuções separadas.
  - litellm: qualquer outro nome (ex.: ollama/llama3.1:8b,
    ollama/deepseek-r1:8b, ollama/mistral-nemo) -- mesmo padrão de 02/03/06.

O prompt closed-book é EXATAMENTE o mesmo do treino: este script importa
SYSTEM_PROMPT_CLOSED_BOOK diretamente de 11_train_qwen_unsloth.py (não
duplica a string), para nunca dessincronizar treino e avaliação de novo.

Geração determinística (greedy / temperature=0) em todos os modelos, para
comparação justa entre execuções.

USO:
    # baselines via Ollama, várias de uma vez
    python src/08_evaluate_llms.py --test-file out/07_splits_out/govbench_br_validado_test.jsonl \
        --models ollama/llama3.1:8b,ollama/deepseek-r1:8b,ollama/mistral-nemo:12b \
        --output-dir out/08_eval_out

    # Qwen3.5 base (execução separada)
    python src/08_evaluate_llms.py --test-file out/07_splits_out/govbench_br_validado_test.jsonl \
        --models qwen_base --base-model unsloth/Qwen3.5-9B --output-dir out/08_eval_out

    # Qwen3.5 fine-tunado (execução separada)
    python src/08_evaluate_llms.py --test-file out/07_splits_out/govbench_br_validado_test.jsonl \
        --models qwen_finetuned --base-model unsloth/Qwen3.5-9B \
        --lora-adapter out/11_finetuning_out/qwen_govbench_lora --output-dir out/08_eval_out

SAÍDA (acumulativa/retomável -- todas as execuções escrevem no mesmo arquivo):
    out/08_eval_out/predictions.jsonl
"""

import argparse
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path

# Ignora checagem do torchvision (não utilizado para inferência de texto)
os.environ["UNSLOTH_SKIP_TORCHVISION_CHECK"] = "1"

# Alias functorch -> torch.functorch para PyTorch 2.x
try:
    import torch.functorch
    sys.modules["functorch"] = torch.functorch
except Exception:
    pass


def extract_final_answer(text: str) -> str:
    """Extrai estritamente a resposta final, removendo o processo de raciocínio
    (<think>...</think> ou blocos textuais de Thinking Process em inglês)."""
    if not text:
        return ""
    text = text.strip()

    # 1. Se contém a tag de fechamento </think> (padrão DeepSeek e Qwen)
    if "</think>" in text:
        return text.split("</think>")[-1].strip()

    # 2. Se a tag <think> foi aberta no início mas não fechou
    if text.startswith("<think>"):
        text = text[7:].strip()

    # 3. Se o modelo produziu thinking em inglês sem tags XML
    if text.startswith(("Thinking Process:", "Here's a thinking process", "The user is asking", "To answer this question")):
        markers = [
            r"\n\n(?=[A-ZÀ-Ú][a-zà-ú]+.*(?:De acordo|A |O |Em |Com base|Para |No |Segundo |Conforme |O Art|A Lei|As |Os |Quanto ))",
            r"\n\n\*\*Resposta:?\*\*\s*",
            r"\n\nResposta:?\s*",
            r"\n\n\*\*Final Answer:?\*\*\s*",
            r"\n\nFinal Answer:?\s*",
        ]
        for marker in markers:
            parts = re.split(marker, text)
            if len(parts) > 1:
                return parts[-1].strip()

    return text.strip()


def load_train_script_constants(script_dir: Path):
    """Importa SYSTEM_PROMPT_CLOSED_BOOK direto de 11_train_qwen_unsloth.py
    (sem duplicar a string) -- import é seguro e leve, pois esse arquivo só
    importa unsloth/torch DENTRO de main(), não no nível de módulo."""
    path = script_dir / "11_train_qwen_unsloth.py"
    spec = importlib.util.spec_from_file_location("train_script_11", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.SYSTEM_PROMPT_CLOSED_BOOK


def load_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# --------------------------------------------------------------------------
# MOTOR 1: unsloth (Qwen3.5 base ou base+LoRA, in-process)
# --------------------------------------------------------------------------

def run_unsloth_inference(model_key: str, items: list, args, system_prompt: str, out_f, done_ids: set):
    from unsloth import FastLanguageModel

    print(f"Carregando {args.base_model} em 4-bit...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_len,
        dtype=None,
        load_in_4bit=True,
        device_map={"": 0},
    )

    if model_key == "qwen_finetuned":
        if not args.lora_adapter:
            raise ValueError("--lora-adapter é obrigatório para --models qwen_finetuned")
        print(f"Carregando adaptador LoRA de {args.lora_adapter}...")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora_adapter)

    FastLanguageModel.for_inference(model)  # modo rápido de inferência do Unsloth

    for i, item in enumerate(items, start=1):
        key = (model_key, item["id"])
        if key in done_ids:
            continue
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": item["pergunta"]},
        ]
        try:
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        inputs = tokenizer(text=prompt_text, return_tensors="pt").to("cuda")

        t0 = time.time()
        try:
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
            )
            gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            raw_resposta = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            resposta = extract_final_answer(raw_resposta)
            error = None
        except Exception as e:
            resposta = None
            error = str(e)
        latency = time.time() - t0

        row = {
            "id": item["id"], "dominio": item["dominio"], "nivel_dificuldade": item["nivel_dificuldade"],
            "model": model_key, "pergunta": item["pergunta"],
            "resposta_referencia": item["resposta_referencia"],
            "chunk_texto": item.get("chunk_texto", []),
            "resposta_gerada": resposta, "latency_s": round(latency, 2), "error": error,
        }
        out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
        out_f.flush()
        print(f"  [{i}/{len(items)}] {item['id']} ({model_key}) -> "
              f"{'OK' if error is None else 'FALHOU: ' + error[:60]}")


# --------------------------------------------------------------------------
# MOTOR 2: litellm (baselines via Ollama ou outra API)
# --------------------------------------------------------------------------

def run_litellm_inference(model_name: str, items: list, args, system_prompt: str, out_f, done_ids: set):
    from litellm import completion

    for i, item in enumerate(items, start=1):
        key = (model_name, item["id"])
        if key in done_ids:
            continue

        t0 = time.time()
        resposta, error = None, None
        for attempt in range(args.max_retries + 1):
            try:
                resp = completion(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": item["pergunta"]},
                    ],
                    temperature=0,
                    max_tokens=args.max_new_tokens,
                    timeout=180,
                )
                raw_resposta = resp["choices"][0]["message"]["content"].strip()
                resposta = extract_final_answer(raw_resposta)
                error = None
                break
            except Exception as e:
                error = str(e)
                time.sleep(1.5 * (attempt + 1))
        latency = time.time() - t0

        row = {
            "id": item["id"], "dominio": item["dominio"], "nivel_dificuldade": item["nivel_dificuldade"],
            "model": model_name, "pergunta": item["pergunta"],
            "resposta_referencia": item["resposta_referencia"],
            "chunk_texto": item.get("chunk_texto", []),
            "resposta_gerada": resposta, "latency_s": round(latency, 2), "error": error,
        }
        out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
        out_f.flush()
        print(f"  [{i}/{len(items)}] {item['id']} ({model_name}) -> "
              f"{'OK' if error is None else 'FALHOU: ' + error[:60]}")


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test-file", required=True)
    parser.add_argument("--models", required=True,
                         help="'qwen_base' / 'qwen_finetuned' (só 1 por execução) OU lista litellm separada por vírgula")
    parser.add_argument("--base-model", default="unsloth/Qwen3.5-9B")
    parser.add_argument("--lora-adapter", default=None, help="obrigatório para --models qwen_finetuned")
    parser.add_argument("--output-dir", default="out/08_eval_out")
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=1700)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true", help="sobrescreve predições anteriores dos modelos solicitados")
    parser.add_argument("--limit", type=int, default=None, help="sanity check com N itens")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    system_prompt = load_train_script_constants(script_dir)

    items = load_jsonl(Path(args.test_file))
    if args.limit:
        items = items[: args.limit]
    print(f"Itens de teste: {len(items)}")

    requested_models = [m.strip() for m in args.models.split(",") if m.strip()]
    unsloth_models = [m for m in requested_models if m in ("qwen_base", "qwen_finetuned")]
    litellm_models = [m for m in requested_models if m not in ("qwen_base", "qwen_finetuned")]

    if len(unsloth_models) > 1:
        raise ValueError(
            "Só um modelo Unsloth (qwen_base OU qwen_finetuned) por execução -- "
            "8GB de VRAM não comporta os dois no mesmo processo com segurança. "
            "Rode em duas chamadas separadas."
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "predictions.jsonl"

    existing = load_jsonl(output_path)
    if args.overwrite:
        kept = [row for row in existing if row.get("model") not in requested_models]
        with open(output_path, "w", encoding="utf-8") as f:
            for row in kept:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        existing = kept
        print(f"Modo --overwrite: {len(existing)} predições de outros modelos preservadas.")

    done_ids = {(row["model"], row["id"]) for row in existing}
    print(f"Já concluído (pulado): {len(done_ids)} pares (modelo, item)")

    with open(output_path, "a", encoding="utf-8") as out_f:
        for m in unsloth_models:
            print(f"\n=== Motor Unsloth: {m} ===")
            run_unsloth_inference(m, items, args, system_prompt, out_f, done_ids)

        for m in litellm_models:
            print(f"\n=== Motor litellm: {m} ===")
            run_litellm_inference(m, items, args, system_prompt, out_f, done_ids)

    print(f"\nConcluído. Saída acumulada em: {output_path}")


if __name__ == "__main__":
    main()
