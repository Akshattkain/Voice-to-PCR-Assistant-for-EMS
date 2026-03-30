"""Training hyperparameters for fine-tuning."""

from pydantic import BaseModel


class TrainingConfig(BaseModel):
    """Configuration for T5/Llama fine-tuning."""

    # Model
    base_model: str = "t5-base"
    max_source_length: int = 512
    max_target_length: int = 1024

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: list[str] = ["q", "v"]

    # Training
    learning_rate: float = 3e-4
    batch_size: int = 8
    num_epochs: int = 10
    warmup_steps: int = 100
    weight_decay: float = 0.01
    fp16: bool = True
    gradient_accumulation_steps: int = 2

    # Data
    data_dir: str = "training/data/processed"
    output_dir: str = "models/t5_pcr_v1"

    # Evaluation
    eval_steps: int = 100
    save_steps: int = 200
    logging_steps: int = 50
