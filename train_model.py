import os
import logging
import json
import pandas as pd
from typing import Optional, List, Dict
import config
from tqdm import tqdm
import subprocess
import yaml
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


class ModelTrainer:
    def __init__(self, dataset_path: str, output_dir: str):
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.framework = config.TRAINING_FRAMEWORK

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Default fine-tuning configuration
        self.fine_tune_config = {
            "fine_tune_type": "lora",  # lora, dora, or full
            "iters": 1000,
            "batch_size": 4,
            "lora_layers": 16,
            "learning_rate": 1e-5,
            "val_batches": 25,
            "steps_per_report": 10,
            "steps_per_eval": 200,
            "max_seq_length": 2048,
            "save_every": 100,
            "grad_checkpoint": False,
            "seed": 42,
            "adapter_path": os.path.join(output_dir, "adapters"),
            "resume_adapter_file": None,
        }

    def prepare_dataset(self) -> tuple[str, str, str, str]:
        """
        Load dataset and split into train/test/valid JSON files
        Returns paths to train.jsonl, test.jsonl, and valid.jsonl
        """
        df = pd.read_csv(self.dataset_path)

        # Create formatted examples with chat template
        examples = []
        for _, row in df.iterrows():
            # Format as chat messages
            messages = [
                {"role": "system", "content": config.EMAIL_CONFIG["system_prompt"]},
                {
                    "role": "user",
                    "content": row["prompt"],
                },
                {"role": "assistant", "content": row["completion"]},
            ]

            examples.append({"messages": messages})

        train_data, test_data = train_test_split(
            examples, test_size=0.2, random_state=42
        )
        train_data, valid_data = train_test_split(
            train_data, test_size=0.1, random_state=42
        )

        # Save to JSONL files
        data_dir = os.path.join(self.output_dir, "data")
        os.makedirs(data_dir, exist_ok=True)

        def save_to_jsonl(data: List[Dict], filename: str) -> str:
            filepath = os.path.join(data_dir, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            return filepath

        train_path = save_to_jsonl(train_data, "train.jsonl")
        test_path = save_to_jsonl(test_data, "test.jsonl")
        valid_path = save_to_jsonl(valid_data, "valid.jsonl")

        return data_dir, train_path, test_path, valid_path

    def train(self, fine_tune_config: Optional[dict] = None):
        """Train model using appropriate framework with given config"""
        if fine_tune_config:
            self.fine_tune_config.update(fine_tune_config)

        if self.framework == "mlx":
            return self._train_mlx()
        else:
            return self._train_transformers()

    def _train_mlx(self):
        """Train using MLX framework with LoRA"""
        try:
            # Prepare dataset in JSONL format
            data_dir, _, _, _ = self.prepare_dataset()
            logging.info(f"Done preparing dataset. Data directory: {data_dir}")

            # Save fine-tuning config
            config_path = os.path.join(self.output_dir, "fine_tune.yaml")
            with open(config_path, "w") as f:
                yaml.dump(self.fine_tune_config, f)
                logging.info(
                    f"Done saving fine-tuning config. Config path: {config_path}"
                )

            # Prepare MLX LoRA training command
            command = [
                # "python",
                # "-m",
                "mlx_lm.lora",
                "--model",
                config.BASE_MODEL,
                "--train",
                "--data",
                data_dir,
                "--config",
                config_path,
            ]

            # Set PYTHONUNBUFFERED for real-time output
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"

            # Run training process
            process = subprocess.Popen(
                command,
                bufsize=1,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                env=env,
            )

            # Stream output
            logging.info("Training started. Streaming output...\n")
            for line in iter(process.stdout.readline, ""):
                logging.info(line.strip())
                yield line.strip()

            # Check return code
            return_code = process.poll()
            if return_code == 0:
                logging.info("Training completed successfully")
                yield "Training completed successfully"
            else:
                logging.error(f"Training failed with error code {return_code}")
                yield f"Training failed with error code {return_code}"

        except Exception as e:
            logging.error(f"Error in MLX training: {str(e)}")
            yield f"Error in MLX training: {str(e)}"

    def load_fine_tuned_model(self, adapter_path: Optional[str] = None):
        """Load the fine-tuned model with adapters"""
        if adapter_path is None:
            adapter_path = self.fine_tune_config["adapter_path"]

        try:
            from mlx_lm import load

            # Load base model with adapters
            model, tokenizer = load(config.BASE_MODEL, adapter_path=adapter_path)

            return model, tokenizer

        except Exception as e:
            logging.error(f"Error loading fine-tuned model: {str(e)}")
            return None, None


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train email response model")
    parser.add_argument("--dataset", required=True, help="Path to dataset CSV")
    parser.add_argument("--output", required=True, help="Output directory for model")

    # Add fine-tuning specific arguments
    parser.add_argument(
        "--fine-tune-type", default="lora", choices=["lora", "dora", "full"]
    )
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lora-layers", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument(
        "--resume-adapter", help="Path to resume training with given adapters"
    )

    args = parser.parse_args()

    # Create trainer
    trainer = ModelTrainer(args.dataset, args.output)

    # Update fine-tuning config from arguments
    fine_tune_config = {
        "fine_tune_type": args.fine_tune_type,
        "iters": args.iterations,
        "batch_size": args.batch_size,
        "lora_layers": args.lora_layers,
        "learning_rate": args.learning_rate,
    }
    if args.resume_adapter:
        fine_tune_config["resume_adapter_file"] = args.resume_adapter

    # Start training
    for output in trainer.train(fine_tune_config):
        print(output)


if __name__ == "__main__":
    main()
