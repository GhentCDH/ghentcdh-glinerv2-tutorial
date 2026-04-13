import argparse
import gc
import json
import torch
from gliner2 import GLiNER2
from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig
from gliner2.training.data import InputExample

DOMAIN_FILES = {
    "date":         "./data/train_date.json",
    "organisation": "./data/train_organisation.json",
    "legal":        "./data/train_legal.json",
    "person":       "./data/train_person.json",
    "share":        "./data/train_share.json",
    "location":     "./data/train_location.json",
}

def load_domain_examples(domain_name: str):
    file_path = DOMAIN_FILES.get(domain_name)
    if not file_path:
        raise ValueError(f"No file configured for domain '{domain_name}'.")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [
        InputExample(text=item["text"], entities=item["entities"])
        for item in data
    ]

def train_domain_adapter(base_model_name, examples, domain_name, output_dir):
    adapter_path = f"{output_dir}/{domain_name}_adapter"
    config = TrainingConfig(
        output_dir=adapter_path,
        experiment_name=f"{domain_name}_domain",
        num_epochs=10,
        batch_size=4,
        gradient_accumulation_steps=4,
        encoder_lr=1e-5,
        task_lr=5e-4,
        use_lora=True,
        lora_r=8,
        lora_alpha=16.0,
        lora_dropout=0.0,
        lora_target_modules=["encoder"],
        save_adapter_only=True,
        eval_strategy="epoch",
        eval_steps=500,
        logging_steps=50,
        fp16=True,
    )
    print(f"\n{'='*60}")
    print(f"Training {domain_name.upper()} adapter")
    print(f"{'='*60}")
    model = GLiNER2.from_pretrained(base_model_name)
    trainer = GLiNER2Trainer(model=model, config=config)
    results = trainer.train(train_data=examples)
    final_path = f"{adapter_path}/final"
    print(f"\n✅ {domain_name.capitalize()} adapter trained!")
    print(f"📁 Saved to: {final_path}/")
    print(f"⏱️  Training time: {results['total_time_seconds']:.2f}s")
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain",     required=True)
    parser.add_argument("--base_model", default="fastino/gliner2-multi-v1")
    parser.add_argument("--output_dir", default="./adapters")
    args = parser.parse_args()
    examples = load_domain_examples(args.domain)
    train_domain_adapter(args.base_model, examples, args.domain, args.output_dir)
