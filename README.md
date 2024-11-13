# LAMBDA (LocalAutoMailBoxDraftAssistant)
An AI-based email automation system that learns from your email style and creates draft responses using MLX.

## Features
- Automatically extracts your email communication patterns from Gmail
- Fine-tunes LLaMA models on your email style using MLX's LoRA implementation
- Creates AI-generated draft responses for unread emails
- Supports both Apple Silicon (MLX) and CUDA GPUs

## Requirements
- Python 3.8+
- Gmail Account with API access
- For Mac: MLX compatible device (M1/M2/M3)
- For others: CUDA compatible GPU

## Installation
1. Clone the repository:
```bash
git clone https://github.com/zycyc/LAMBDA.git
cd LAMBDA
```

2. Install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. Enable Gmail API:
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project
   - Enable Gmail API
   - Create OAuth 2.0 credentials
   - Download credentials as `credentials.json` and place in project root

## Usage

### 1. Create Training Dataset
```bash
python create_dataset.py --output email_dataset.csv
```

### 2. Train Model
```bash
python train_model.py --dataset email_dataset.csv --output model_output
```

### 3. Run Email Bot
```bash
python lambda_bot.py
```

The bot will:
- Check for unread emails every 5 minutes (adjustable with --interval)
- Generate responses using your fine-tuned model
- Create draft responses in Gmail
- Skip threads that already have drafts

## Project Structure
```
LAMBDA/
├── README.md
├── requirements.txt
├── config.py              # Configuration settings
├── create_dataset.py      # Gmail data extraction
├── train_model.py         # Model fine-tuning
├── lambda_bot.py          # Email bot implementation
└── gmail_utils.py         # Gmail API utilities
```

## Configuration
Edit `config.py` to customize:
- Model selection
- Training parameters
- Email settings
- Response templates

## License
MIT License - see LICENSE file for details