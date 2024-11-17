import platform

# System detection
IS_MAC = platform.system() == "Darwin" and platform.machine() == "arm64"

# Model configurations
if IS_MAC:
    BASE_MODEL = "mlx-community/Llama-3.2-3B-Instruct"
    TRAINING_FRAMEWORK = "mlx"
else:
    BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
    TRAINING_FRAMEWORK = "transformers"

# Email configurations
USER_NAME = "Alan"
EMAIL_CONFIG = {
    "bot_signature": "\n\n[This is an AI-generated draft response from LAMBDA. Please carefully review the response before sending it.]",
    "bot_label": "LAMBDA",
    "interval": 300,  # in seconds (i.e., 5 minutes)
    "max_response_length": 1000,
    "max_thread_length": 1000,
    "system_prompt": "You are {user_name}. You will receive an email and your job is to write a response to the email as if you were {user_name}.",
    "format_prompt": """
    "Take the provided response and reformat it for proper email communication. Ensure the following:
    1. Add proper punctuation and capitalization to maintain professionalism.
    2. Separate ideas into clearly distinct paragraphs for better readability.
    3. Do not change the content of the response unless necessary to improve readability.
    Return only the reformatted email response, with no additional text or commentary or email title.""",
    "ignore_labels": {
        "CATEGORY_SOCIAL",
        "CATEGORY_UPDATES",
        "CATEGORY_FORUMS",
        "CATEGORY_PROMOTIONS",
    },
}

# Prompt template for training
PROMPT_TEMPLATE = """
Conversation history: {context}

Latest message:
From: {reply_from}
To: {reply_to}
Cc: {reply_cc}
Subject: {reply_subject}

{original_content}
"""

# Response template for inference
RESPONSE_TEMPLATE = """{bot_signature}

{generated_response}

"""

# Default fine-tuning configuration for MLX
FINE_TUNE_CONFIG_MLX = {
    "fine_tune_type": "lora",
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
    "resume_adapter_file": None,
}

# Configuration for CUDA/Transformers training
FINE_TUNE_CONFIG_CUDA = {
    "disable_tqdm": False,
    "batch_size": 4,
    "num_epochs": 3,
    "learning_rate": 1e-5,
    "max_seq_length": 2048,
    "warmup_ratio": 0.05,
    "save_strategy": "epoch",
    "evaluation_strategy": "epoch",
    "save_total_limit": 1,
    "lora_config": {
        "r": 8,
        "lora_alpha": 16,
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    },
    "quantization": {
        "load_in_4bit": True,
        "bnb_4bit_compute_dtype": "float16",
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
    },
}

# Select appropriate config based on framework
FINE_TUNE_CONFIG = (
    FINE_TUNE_CONFIG_MLX if TRAINING_FRAMEWORK == "mlx" else FINE_TUNE_CONFIG_CUDA
)
