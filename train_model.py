import os
import logging
import json
import pandas as pd
from typing import Optional, List, Dict
import config
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

        # Initialize fine-tuning configuration from config.py
        self.fine_tune_config = config.FINE_TUNE_CONFIG.copy()
        self.fine_tune_config["adapter_path"] = os.path.join(output_dir, "adapters")

    def prepare_dataset(self) -> tuple[str, str, str, str]:
        """
        Load dataset and split into train/test/valid JSON files
        Returns paths to train.jsonl, test.jsonl, and valid.jsonl
        """
        df = pd.read_csv(self.dataset_path)

        # # for now, use only the first 10 rows
        df = df.head(5)

        # Create formatted examples with chat template
        examples = []
        for _, row in df.iterrows():
            # Format as chat messages
            messages = [
                {
                    "role": "system",
                    "content": str(
                        config.EMAIL_CONFIG["system_prompt"].format(
                            user_name=config.USER_NAME
                        )
                    ),
                },
                {"role": "user", "content": str(row["prompt"])},
                {"role": "assistant", "content": str(row["completion"])},
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

    def _train_transformers(self):
        """Train using HuggingFace Transformers with QLoRA"""
        try:
            # Prepare dataset in JSONL format
            data_dir, train_path, _, valid_path = self.prepare_dataset()
            logging.info(f"Done preparing dataset. Data directory: {data_dir}")

            # Save fine-tuning config
            config_path = os.path.join(self.output_dir, "fine_tune.yaml")
            with open(config_path, "w") as f:
                yaml.dump(self.fine_tune_config, f)
                logging.info(
                    f"Done saving fine-tuning config. Config path: {config_path}"
                )
        except Exception as e:
            logging.error(f"Error in preparing dataset: {str(e)}")
            yield f"Error in preparing dataset: {str(e)}"

        # try:
        from transformers import (
            AutoModelForCausalLM,
            TrainingArguments,
            BitsAndBytesConfig,
        )
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from trl import SFTTrainer
        import torch
        from huggingface_hub import login
        from dotenv import load_dotenv
        from datasets import load_dataset

        # Check for CUDA availability first
        if not torch.cuda.is_available():
            logging.error("CUDA is not available. This training requires a GPU.")
            return "Training requires CUDA-enabled GPU."

        # Try to load token from .env file
        load_dotenv()
        hf_token = os.environ.get("HF_TOKEN")

        # If token not found, prompt user and save to .env
        if not hf_token:
            hf_token = input("Please enter your HuggingFace token: ")
            with open(".env", "a") as f:
                f.write(f"\nHF_TOKEN={hf_token}")
            os.environ["HF_TOKEN"] = hf_token
            logging.info("Token saved to .env file")

        login(token=hf_token)

        # Convert deprecated quantization config to BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        # Load base model with updated quantization config
        model = AutoModelForCausalLM.from_pretrained(
            config.BASE_MODEL,
            quantization_config=quantization_config,
            device_map="auto",
        )

        # Prepare model for k-bit training
        model = prepare_model_for_kbit_training(model)

        # Configure LoRA
        lora_config = LoraConfig(**self.fine_tune_config["lora_config"])

        # Get PEFT model
        model = get_peft_model(model, lora_config)

        # Setup training arguments
        training_args = TrainingArguments(
            disable_tqdm=self.fine_tune_config["disable_tqdm"],
            output_dir=self.fine_tune_config["adapter_path"],
            num_train_epochs=self.fine_tune_config["num_epochs"],
            per_device_train_batch_size=self.fine_tune_config["batch_size"],
            learning_rate=self.fine_tune_config["learning_rate"],
            save_strategy=self.fine_tune_config["save_strategy"],
            evaluation_strategy=self.fine_tune_config["evaluation_strategy"],
            save_total_limit=self.fine_tune_config["save_total_limit"],
            warmup_ratio=self.fine_tune_config["warmup_ratio"],
            lr_scheduler_type="constant",
        )

        # Load training dataset
        train_dataset = load_dataset("json", data_files=train_path, split="train")
        print("train dataset", train_dataset)
        eval_dataset = load_dataset("json", data_files=valid_path, split="train")

        # Initialize SFT trainer
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            max_seq_length=self.fine_tune_config["max_seq_length"],
        )

        # Train model
        logging.info("Training started...")
        trainer.train()

        # Save trained model
        trainer.save_model()

        logging.info("Training completed successfully")
        yield "Training completed successfully"

        # except Exception as e:
        #     logging.error(f"Error in Transformers training: {str(e)}")
        #     yield f"Error in Transformers training: {str(e)}"


def find_latest_adapter_file(adapter_path: str) -> Optional[str]:
    """
    Find the most recently created adapter file matching pattern '00*adapters.safetensors'

    Args:
        adapter_path: Directory path containing adapter files

    Returns:
        Optional[str]: Path to the latest adapter file, or None if no files found
    """
    if not os.path.exists(adapter_path):
        return None

    # Find all files matching the pattern
    adapter_files = [
        f
        for f in os.listdir(adapter_path)
        if f.startswith("00") and f.endswith("adapters.safetensors")
    ]

    if not adapter_files:
        return None

    # Get full path and creation time for each file
    files_with_time = [
        (os.path.join(adapter_path, f), os.path.getctime(os.path.join(adapter_path, f)))
        for f in adapter_files
    ]

    # Return the path of the most recently created file
    return max(files_with_time, key=lambda x: x[1])[0]


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train email response model")
    parser.add_argument("--dataset", required=True, help="Path to dataset CSV")
    parser.add_argument("--output", required=True, help="Output directory for model")

    args = parser.parse_args()

    # Create trainer
    trainer = ModelTrainer(args.dataset, args.output)

    # if adapter exists, ask if want to resume training or start over or do nothing
    def _train():
        for output in trainer.train(trainer.fine_tune_config):
            logging.info(output)

    resume_adapter_file = find_latest_adapter_file(
        trainer.fine_tune_config["adapter_path"]
    )
    if resume_adapter_file:
        response = input(
            f"Found latest adapter file: {resume_adapter_file}. Do you want to resume training (r) or start over (s) or do nothing (n)? (r/s/n): "
        )
        if response.lower() == "r":
            trainer.fine_tune_config["resume_adapter_file"] = resume_adapter_file
            _train()
        elif response.lower() == "s":
            logging.info("Starting training from scratch.")
            _train()
        elif response.lower() == "n":
            logging.info("Do nothing in train_model.py.")
    else:
        logging.info("No existing adapter found. Starting training from scratch.")
        _train()


if __name__ == "__main__":
    main()
