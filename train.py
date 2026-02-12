# train.py — Piri Fine-tuning Script
"""
Piri model fine-tuning aracı.
Qwen2.5-0.5B-Instruct modelini özel veriyle fine-tune eder.
ChatML formatında Bağlam/Soru/Cevap örnekleri kullanır.

AKIS Platform tarafından geliştirilmiştir.

Kullanım:
    python train.py                    # Varsayılan ayarlarla eğit
    python train.py --epochs 1         # Tek epoch (hızlı test)
    python train.py --skip-train       # Sadece modeli indir, eğitme
"""
import argparse
import json
import os
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = "./model"
DATA_PATH = "data/rag_train.jsonl"


def load_chat_data(path: str) -> list:
    """ChatML formatında eğitim verisini yükle."""
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    print(f"  {len(examples)} eğitim örneği yüklendi.")
    return examples


def main():
    parser = argparse.ArgumentParser(description="Piri Fine-tuning")
    parser.add_argument("--epochs", type=int, default=2, help="Epoch sayısı")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch boyutu")
    parser.add_argument("--max-length", type=int, default=512, help="Max token uzunluğu")
    parser.add_argument("--lr", type=float, default=1e-5, help="Öğrenme oranı")
    parser.add_argument("--skip-train", action="store_true", help="Sadece modeli indir")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  Piri — Model Fine-tuning")
    print(f"  AKIS Platform")
    print(f"{'='*60}")
    print(f"  Base Model: {MODEL_NAME}")
    print(f"  Çıktı: {OUTPUT_DIR}")

    # 1. Tokenizer ve Model yükle
    print(f"\n[1/5] Model indiriliyor / yükleniyor...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    print(f"  Model boyutu: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M parametre")

    if args.skip_train:
        print(f"\n[SKIP] Eğitim atlandı. Model indiriliyor...")
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"  Model '{OUTPUT_DIR}' klasörüne kaydedildi.")
        return

    # 2. Eğitim verisini yükle
    print(f"\n[2/5] Eğitim verisi yükleniyor...")
    if not os.path.exists(DATA_PATH):
        print(f"  HATA: {DATA_PATH} bulunamadı!")
        print(f"  Önce eğitim verisi oluşturun veya --skip-train kullanın.")
        return

    raw_data = load_chat_data(DATA_PATH)
    dataset = Dataset.from_list(raw_data)

    # 3. Tokenize — ChatML template uygula
    print(f"\n[3/5] Tokenize ediliyor (max_length={args.max_length})...")

    def tokenize_chat(examples):
        texts = []
        for messages in examples["messages"]:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)

        tokenized = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=args.max_length,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized_dataset = dataset.map(
        tokenize_chat,
        batched=True,
        remove_columns=dataset.column_names,
    )
    print(f"  {len(tokenized_dataset)} örnek tokenize edildi.")

    # 4. Data Collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # 5. Eğitim
    print(f"\n[4/5] Eğitim başlıyor...")
    print(f"  Epoch: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        save_steps=200,
        logging_steps=10,
        learning_rate=args.lr,
        weight_decay=0.01,
        fp16=False,
        eval_strategy="no",
        save_total_limit=2,
        report_to="none",
        gradient_accumulation_steps=4,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    print("  Eğitim tamamlandı!")

    # 5. Kaydet
    print(f"\n[5/5] Model kaydediliyor...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"  Model '{OUTPUT_DIR}' klasörüne kaydedildi.")
    print(f"\n{'='*60}")
    print(f"  Piri — Sonraki adımlar:")
    print(f"  1. python ingest.py     # Knowledge base indeksle")
    print(f"  2. python evaluate.py   # Kalite testi çalıştır")
    print(f"  3. uvicorn main:app     # Piri API başlat")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
