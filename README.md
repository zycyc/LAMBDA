# LAMBDA (LocalAutoMailBoxDraftAssistant)
An AI-based email automation system that learns from your email style and creates draft responses using MLX.

## Features
- Automatically extracts your email communication patterns from Gmail
- Fine-tunes LLaMA models on your email style using LoRA
- Creates AI-generated draft responses for unread emails
- Supports both Apple Silicon (MLX) and CUDA GPUs

## Requirements
- Python 3.10+
- Gmail Account with API access
- For Mac: MLX compatible device (M1/M2/M3/M4/...)
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

3. Enable Gmail API and Setup Credentials:
   1. Go to [Google Cloud Console](https://console.cloud.google.com/)
   2. Create a new project:
      - Click on the project dropdown at the top of the page
      - Click "New Project"
      - Enter a project name (e.g., "LAMBDA Email Assistant")
      - Click "Create"
      - Wait for the project to be created and click "SELECT PROJECT"
   
   3. Enable Gmail API:
      - Click on the hamburger menu (☰) in the top-left corner
      - Navigate to "APIs & Services" > "Library"
      - Search for "Gmail API"
      - Click on "Gmail API" in the results
      - Click "Enable"
   
   4. Configure OAuth consent screen:
      - Go to "APIs & Services" > "OAuth consent screen"
      - Select "External" user type (unless you're in an organization)
      - Click "Create"
      - Fill in the required fields:
        * App name: "LAMBDA Email Assistant"
        * User support email: your email
        * Developer contact information: your email
      - Click "Save and Continue"
      - On "Scopes" page
        - Click "ADD OR REMOVE SCOPES"
        - Search for "Gmail API" and select "Read, compose, and send emails from your Gmail account"
        - Click "UPDATE" and then "SAVE AND CONTINUE"
      - On "Test users" page, click "ADD USER" and enter your email, then click "SAVE AND CONTINUE"
      - Click "Back to Dashboard"
   
   5. Create OAuth 2.0 credentials:
      - Go to "APIs & Services" > "Credentials"
      - Click "Create Credentials" at the top
      - Select "OAuth client ID"
      - Choose "Desktop app" as the application type
      - Name it "LAMBDA Desktop Client"
      - Click "Create"
      
   6. Download credentials:
      - In the popup that appears, click "Download" (or download from the credentials page)
      - Rename the downloaded file to `credentials.json`
      - Move the file to your LAMBDA project root directory

   Note: When you first run the application, it will open a browser window asking you to authorize the application. This is normal and only needs to be done once.

   Troubleshooting:
   - If you get a "Google hasn't verified this app" screen, click "Advanced" and then "Go to [Your App Name] (unsafe)"
   - This warning appears because you're using a development version of the app
   - The app only accesses your own Gmail account based on the permissions you grant

## Usage

Run the LAMBDA interface:
```bash
python lambda.py
```

The interactive menu will guide you through the following options:
1. Create/Update training dataset
2. Train model
3. Run email bot
4. Run complete workflow (all steps)
5. Exit

On first run, you'll be prompted to enter your name if not already configured. This name will be used by the AI to personalize responses.

The bot will:
- Check for unread emails every 5 minutes (adjustable in config.py)
- Generate responses using your fine-tuned model
- Create draft responses in Gmail
- Skip threads that already have drafts

## Project Structure
```
LAMBDA/
├── README.md
├── requirements.txt
├── config.py              # Configuration settings
├── lambda.py              # Main interface script
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

## TODO
- [ ] Windows support (now only tested on Mac)
- [ ] RAG support for personal knowledge base
- [ ] ...

## License
MIT License - see LICENSE file for details