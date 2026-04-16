# GLiNER2 LoRA Hagiographics Notebook

This folder contains a complete workflow for training and using GLiNER2 LoRA adapters on hagiographic texts.

Main notebook:
- `GLiNER2_LoRA_hagiographics.ipynb`

What the notebook covers:
- Filter and clean Label Studio exports
- Convert annotations into GLiNER2 training format
- Split data into train/validation/test
- Train LoRA adapters (single-domain and multi-domain)
- Evaluate adapters and compare F1 scores
- Run batch inference on new texts
- Export predictions to Label Studio pre-annotation JSON

## Folder structure (important paths)

- `gliner_schema_hagiographics.json`: schema used for entity definitions
- `gliner2_training_data*.json`: intermediate training data files
- `data_single/`, `data_multi/`: train/val/test splits
- `adapter_single_*`, `adapter_multi*`: trained adapter checkpoints
- `test_texts/`: input texts for batch inference

## Requirements

Install dependencies from this folder:

```bash
pip install -r requirements.txt
```

If you run training on GPU, install a CUDA-compatible PyTorch build first, then install the rest:

```bash
# Example only: choose the command that matches your CUDA version from pytorch.org
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## Quick start

1. Open `GLiNER2_LoRA_hagiographics.ipynb`.
2. Select the environment where you installed `requirements.txt`.
3. Run the notebook from top to bottom the first time.
4. For later experiments, jump to the section you need:
   - Data preprocessing
   - Single-domain finetuning
   - Multi-domain finetuning
   - Batch processing

## Notes

- The notebook imports helper functions from `../gliner_to_labelstudio.py`, so keep this repository layout unchanged.
- Keep label names consistent between:
  - your schema (`gliner_schema_hagiographics.json`)
  - training data labels
  - evaluation/inference labels
- Adapter output directories can become large; clean old checkpoints if needed.

## Reproducibility tips

- Keep train/val/test split settings fixed when comparing experiments.
- Use descriptive adapter output folder names per run.
- Record the base model and schema version for each experiment.
