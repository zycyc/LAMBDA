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
USER_NAME = ""
EMAIL_CONFIG = {
    "bot_signature": "\n\n[This is an AI-generated draft response from LAMBDA. Please carefully review the response before sending it.]",
    "bot_label": "LAMBDA",
    "interval": 300,  # in seconds (i.e., 5 minutes)
    "max_response_length": 1000,
    "max_thread_length": 1000,
    "system_prompt": "You are {USER_NAME}. You will receive an email and your job is to write a response to the email as if you were {USER_NAME}.",
    "format_prompt": "Please add proper punctuation and capitalization, and separate paragraphs properly. Please only return the response, no other text.",
}

# Prompt template for training
PROMPT_TEMPLATE = """Email Thread:
{email_thread}

Write a professional response for the following:
From: {reply_from}
To: {reply_to}
Cc: {reply_cc}
Subject: {reply_subject}
"""

# Response template for inference
RESPONSE_TEMPLATE = """{bot_signature}

{generated_response}

"""

# Default fine-tuning configuration
FINE_TUNE_CONFIG = {
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
