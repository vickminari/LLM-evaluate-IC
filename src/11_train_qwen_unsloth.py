#!/usr/bin/env python3
"""
11_train_qwen_unsloth.py
--------------------------------------
Fine-Tuning (QLoRA 4-bit) do Qwen3.5-9B via Unsloth. RTX 4070 8GB / WSL2.

  1. --val-fraction: reserva uma fração do TREINO (não do teste) para
     eval_loss durante o treinamento -- útil para a curva de validação no
     relatório e para detectar overfitting.
  2. MetricsLoggerCallback: grava loss/lr/grad_norm/epoch a cada
     logging_steps (e eval_loss quando houver validação) em JSONL + CSV,
     e gera um gráfico da curva de loss ao final. Nenhuma dependência de
     W&B/TensorBoard -- tudo em arquivo local, pronto para a redação.
  3. Aviso explícito no início da execução sobre QLoRA em Qwen3.5 (para
     ficar registrado no log de toda run).

Entrada:
  - 07_splits_out/govbench_br_validado_train.jsonl

Saída (em --output-dir):
  - qwen_govbench_lora/          adaptador LoRA + tokenizer
  - metrics/train_log.jsonl      1 linha por passo de logging
  - metrics/train_log.csv        mesma coisa em CSV
  - metrics/loss_curve.png       gráfico train/eval loss
  - metrics/run_summary.json     hiperparâmetros + métricas finais

Uso:
  python 11_train_qwen_unsloth.py --epochs 3 --format both --val-fraction 0.1
"""

import argparse
import json
import os
from pathlib import Path
import sys
import types

import importlib.machinery

# Ignora checagem do torchvision (não utilizado para LLM de texto)
os.environ["UNSLOTH_SKIP_TORCHVISION_CHECK"] = "1"
# Desativa offloading de gradientes para a RAM (força processamento 100% GPU VRAM)
os.environ["UNSLOTH_OFFLOAD_GRADIENTS"] = "0"

# Alias functorch -> torch.functorch para PyTorch 2.x
try:
    import torch.functorch
    sys.modules["functorch"] = torch.functorch
except Exception:
    pass

# Stub de compatibilidade para o torchvision no PyTorch dev/nightly
try:
    import torchvision
    # Testa se o operador C++ de nms funciona sem disparar exceção
    torchvision._meta_registrations
except Exception:
    tv = types.ModuleType("torchvision")
    tv.__spec__ = importlib.machinery.ModuleSpec("torchvision", loader=None)
    tvt = types.ModuleType("torchvision.transforms")
    tvt.__spec__ = importlib.machinery.ModuleSpec("torchvision.transforms", loader=None)
    class _InterpolationModeMeta(type):
        def __getattr__(cls, name):
            return name.lower().replace("_", "-")

    class InterpolationMode(metaclass=_InterpolationModeMeta):
        NEAREST = "nearest"
        NEAREST_EXACT = "nearest-exact"
        BILINEAR = "bilinear"
        BICUBIC = "bicubic"
        BOX = "box"
        HAMMING = "hamming"
        LANCZOS = "lanczos"

    tvt.InterpolationMode = InterpolationMode
    tv.transforms = tvt
    sys.modules["torchvision"] = tv
    sys.modules["torchvision.transforms"] = tvt

SYSTEM_PROMPT_RAG = (
    "Você é um assistente especializado na administração pública e legislação brasileira. "
    "Responda à pergunta do usuário de forma precisa, completa e estritamente fundamentada "
    "no contexto fornecido."
)

SYSTEM_PROMPT_CLOSED_BOOK = (
    "Você é um assistente especializado na administração pública e legislação brasileira. "
    "Responda à pergunta do usuário de forma precisa e completa, com base no seu "
    "conhecimento sobre a legislação e os documentos oficiais brasileiros."
)


def format_chatml_sample(item: dict, tokenizer, mode: str) -> str:
    """Converte um item do GovBench-BR para ChatML/Qwen, no formato 'rag'
    (contexto explícito no prompt) ou 'closed_book' (só a pergunta).

    IMPORTANTE: o mesmo par (system_prompt, presença/ausência de contexto)
    usado aqui DEVE ser reproduzido exatamente no script de avaliação
    (3.3.2) para cada protocolo correspondente -- formato de treino e
    formato de avaliação precisam bater, senão reintroduzimos o mesmo tipo
    de mismatch que este patch corrige.
    """
    if mode == "rag":
        context_chunks = item.get("chunk_texto", [])
        if isinstance(context_chunks, list):
            context_str = "\n\n".join(
                f"[Trecho {idx+1}]: {chunk}" for idx, chunk in enumerate(context_chunks)
            )
        else:
            context_str = str(context_chunks)
        user_content = f"Contexto:\n{context_str}\n\nPergunta:\n{item['pergunta']}"
        system_prompt = SYSTEM_PROMPT_RAG
    elif mode == "closed_book":
        user_content = item["pergunta"]
        system_prompt = SYSTEM_PROMPT_CLOSED_BOOK
    else:
        raise ValueError(f"modo desconhecido: {mode}")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": item["resposta_referencia"]},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def build_dataset(raw_items: list, tokenizer, fmt: str) -> list:
    modes = ["rag", "closed_book"] if fmt == "both" else [fmt]
    rows = []
    for item in raw_items:
        for mode in modes:
            rows.append({
                "text": format_chatml_sample(item, tokenizer, mode),
                "source_id": item["id"],
                "format": mode,
            })
    return rows


class MetricsLoggerCallback:
    """Callback leve para transformers.Trainer: grava cada dict de log
    (train e eval) em memória e persiste em JSONL/CSV/PNG ao final. Feito
    para não depender de W&B/TensorBoard -- só arquivos locais."""

    def __init__(self, output_dir: Path):
        from transformers import TrainerCallback

        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history = []

        outer = self

        class _Inner(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs is None:
                    return
                row = dict(logs)
                row["step"] = state.global_step
                row["epoch_at_log"] = logs.get("epoch", state.epoch)
                outer.history.append(row)

        self._callback_instance = _Inner()

    def as_hf_callback(self):
        return self._callback_instance

    def save(self, extra_summary: dict = None):
        import csv

        jsonl_path = self.output_dir / "train_log.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for row in self.history:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        all_keys = sorted({k for row in self.history for k in row.keys()})
        csv_path = self.output_dir / "train_log.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            for row in self.history:
                writer.writerow(row)

        train_losses = [(r["step"], r["loss"]) for r in self.history if "loss" in r]
        eval_losses = [(r["step"], r["eval_loss"]) for r in self.history if "eval_loss" in r]

        plot_path = self.output_dir / "loss_curve.png"
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            plt.figure(figsize=(8, 5))
            if train_losses:
                xs, ys = zip(*train_losses)
                plt.plot(xs, ys, label="train_loss", marker="o", markersize=3)
            if eval_losses:
                xs, ys = zip(*eval_losses)
                plt.plot(xs, ys, label="eval_loss", marker="s", markersize=4)
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.title("Qwen3.5-9B + LoRA/GovBench-BR — curva de treino")
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(plot_path, dpi=150)
            plt.close()
        except Exception as e:
            print(f"[aviso] não consegui gerar o gráfico de loss: {e}")
            plot_path = None

        summary = {
            "n_log_points": len(self.history),
            "final_train_loss": train_losses[-1][1] if train_losses else None,
            "final_eval_loss": eval_losses[-1][1] if eval_losses else None,
            "min_train_loss": min((v for _, v in train_losses), default=None),
            "min_eval_loss": min((v for _, v in eval_losses), default=None),
        }
        if extra_summary:
            summary.update(extra_summary)
        with open(self.output_dir / "run_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"\nMétricas salvas em:")
        print(f"  -> {jsonl_path}")
        print(f"  -> {csv_path}")
        if plot_path:
            print(f"  -> {plot_path}")
        print(f"  -> {self.output_dir / 'run_summary.json'}")
        return summary


def main():
    parser = argparse.ArgumentParser(description="Fine-Tuning do Qwen3.5-9B com Unsloth.")
    parser.add_argument("--model", default="unsloth/Qwen3.5-9B", help="Repo HF do modelo base")
    parser.add_argument("--train-file", default="out/07_splits_out/govbench_br_validado_train.jsonl")
    parser.add_argument("--output-dir", default="out/11_finetuning_out")
    parser.add_argument("--format", choices=["rag", "closed_book", "both"], default="closed_book",
                         help="Formato dos exemplos de treino")
    parser.add_argument("--val-fraction", type=float, default=0.1,
                         help="Fração do TREINO reservada para eval_loss (0 desativa)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=8, help="Padrão 8 para caber em 8GB VRAM sem offload")
    parser.add_argument("--limit", type=int, default=None, help="Sanity check com N itens")
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    from unsloth import FastLanguageModel, is_bfloat16_supported
    try:
        import unsloth_zoo.fused_losses.cross_entropy_loss as _ce
        _orig_get_chunk_multiplier = _ce._get_chunk_multiplier
        def _safe_get_chunk_multiplier(vocab_size, target_gb=None):
            if target_gb is None:
                try:
                    import torch
                    free, total = torch.cuda.mem_get_info(0)
                    free_gb = (free / (1024 ** 3)) * 0.5
                    target_gb = min(max(free_gb, 0.25), 4.0)
                except Exception:
                    target_gb = 0.25
            return _orig_get_chunk_multiplier(vocab_size, target_gb=target_gb)
        _ce._get_chunk_multiplier = _safe_get_chunk_multiplier
    except Exception:
        pass

    from datasets import Dataset
    from trl import SFTTrainer
    from transformers import TrainingArguments

    print("==========================================================")
    print("Iniciando Fine-Tuning do Qwen3.5-9B com Unsloth (SFT/QLoRA)")
    print("==========================================================")
    print(f"Modelo Base:          {args.model}")
    print(f"Formato de treino:    {args.format}"
          f"{' (dataset será dobrado)' if args.format == 'both' else ''}")
    print(f"Fração de validação:  {args.val_fraction}")
    print("[AVISO] A documentação oficial do Unsloth recomenda NÃO usar QLoRA (4-bit) "
          "em modelos Qwen3.5, por diferenças de quantização acima do normal. Usamos "
          "QLoRA aqui por restrição de hardware (RTX 4070 8GB; LoRA bf16 do Qwen3.5-9B "
          "precisa de ~22GB). Documentar isso como limitação no relatório.")
    print("==========================================================\n")

    print("1. Carregando modelo base quantizado em 4-bit...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_len,
        dtype=None,
        load_in_4bit=True,
        device_map="cuda:0",
    )

    print("2. Injetando adaptadores LoRA...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_r,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        use_rslora=False,
        loftq_config=None,
    )

    print(f"3. Carregando e formatando dados de treino a partir de {args.train_file}...")
    train_path = Path(args.train_file)
    with open(train_path, "r", encoding="utf-8") as f:
        raw_items = [json.loads(line) for line in f if line.strip()]

    if args.limit:
        raw_items = raw_items[: args.limit]
        print(f"   -> Modo de Teste Ativo: {len(raw_items)} itens.")

    formatted_rows = build_dataset(raw_items, tokenizer, args.format)
    print(f"   -> {len(raw_items)} itens -> {len(formatted_rows)} exemplos formatados "
          f"({args.format}).")

    full_dataset = Dataset.from_list(formatted_rows)

    eval_dataset = None
    if args.val_fraction and args.val_fraction > 0:
        split = full_dataset.train_test_split(test_size=args.val_fraction, seed=args.seed)
        train_dataset, eval_dataset = split["train"], split["test"]
        print(f"   -> Split interno: {len(train_dataset)} treino / {len(eval_dataset)} validação.")
    else:
        train_dataset = full_dataset

    out_dir_path = Path(args.output_dir)
    lora_out_dir = out_dir_path / "qwen_govbench_lora"
    checkpoint_dir = out_dir_path / "checkpoints"
    metrics_dir = out_dir_path / "metrics"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    metrics_logger = MetricsLoggerCallback(metrics_dir)

    print("4. Configurando SFTTrainer...")
    training_args = TrainingArguments(
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_steps=10,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=5,
        optim="paged_adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=args.seed,
        output_dir=str(checkpoint_dir),
        report_to="none",
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=20 if eval_dataset is not None else None,
        per_device_eval_batch_size=args.batch_size,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_len,
        dataset_num_proc=2,
        packing=False,
        args=training_args,
        callbacks=[metrics_logger.as_hf_callback()],
    )

    print("\n5. Executando o treinamento SFT...")
    trainer_stats = trainer.train()

    print("\nTreinamento concluído.")
    print(f"   - Perda final (Train Loss): {trainer_stats.training_loss:.4f}")
    print(f"   - Tempo total de treino:     {trainer_stats.metrics.get('train_runtime', 0):.2f}s")

    print(f"\n6. Salvando adaptador LoRA e tokenizer em: {lora_out_dir}")
    model.save_pretrained(str(lora_out_dir))
    tokenizer.save_pretrained(str(lora_out_dir))

    metrics_logger.save(extra_summary={
        "model": args.model,
        "format": args.format,
        "epochs": args.epochs,
        "lr": args.lr,
        "lora_r": args.lora_r,
        "max_seq_len": args.max_seq_len,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "n_train_examples": len(train_dataset),
        "n_eval_examples": len(eval_dataset) if eval_dataset is not None else 0,
        "train_runtime_s": trainer_stats.metrics.get("train_runtime", 0),
        "quantization": "4-bit QLoRA (bf16 recomendado pelo Unsloth para Qwen3.5 "
                         "não coube em 8GB VRAM -- ver aviso no início do log)",
    })

    print(f"\nProcesso de Fine-Tuning finalizado. Adaptador em: {lora_out_dir}")


if __name__ == "__main__":
    main()
