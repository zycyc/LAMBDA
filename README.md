![LAMBDA](assets/LAMBDA_banner.png)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/zycyc/LAMBDA/graphs/commit-activity)
[![GitHub issues](https://img.shields.io/github/issues/zycyc/LAMBDA)](https://github.com/zycyc/LAMBDA/issues)
[![GitHub stars](https://img.shields.io/github/stars/zycyc/LAMBDA)](https://github.com/zycyc/LAMBDA/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/zycyc/LAMBDA)](https://github.com/zycyc/LAMBDA/network)
[![GitHub last commit](https://img.shields.io/github/last-commit/zycyc/LAMBDA)](https://github.com/zycyc/LAMBDA/commits/main)

# LAMBDA (LocalAutoMailBoxDraftAssistant)
A local AI-powered email automation system that learns from your email style and creates draft responses for every unread email in your (Gmail) inbox.


https://github.com/user-attachments/assets/e9ffa491-7343-41ae-9cfa-3cca43dad7fb



- Save time replying to emails -- **JUST Open the draft and Send**
- Keep all data and model to yourself -- **No cloud, no tracking**
- Something everyone can use everyday -- **Set up in less than 10 minutes and forget about it**

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
1. **Clone the repository:**
```bash
git clone https://github.com/zycyc/LAMBDA.git
cd LAMBDA
```

2. **Install dependencies:**

For Mac (Apple Silicon) and Linux:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For Windows:
```powershell
python -m venv .venv
Set-ExecutionPolicy Unrestricted -Scope Process
.venv\Scripts\activate
pip install -r requirements.txt
```

Note: If you encounter any issues with PyTorch installation:
- For Windows, you can manually install PyTorch with CUDA using:
  ```powershell
  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
  ```
- For Linux, you can manually install PyTorch using:
  ```bash
  pip3 install torch torchvision torchaudio
  ```

3. **Enable Gmail API and Setup Credentials (only takes about 5 minutes):**
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


## Usage

Run the LAMBDA interface:
```bash
python lambda.py
```
or run the bot in background once you're satisfied with everything:
```bash
nohup python lambda_bot.py &> lambda_bot.log &
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
├── blacklist.txt          # Sender blacklist to ignore
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
- Training hyperparameters
- Email settings
- Response templates

## Email Filtering
LAMBDA provides multiple ways to filter which emails get processed:

### 1. Gmail Labels (Automatic)
By default, LAMBDA skips emails with the following Gmail labels:
- CATEGORY_SOCIAL (social media notifications)
- CATEGORY_UPDATES (automatic updates)
- CATEGORY_FORUMS (forum posts)
- CATEGORY_PROMOTIONS (marketing emails)

### 2. Sender Blacklist (Manual)
You can manually specify which senders to ignore by adding their email addresses to `blacklist.txt`:
1. Open `blacklist.txt` in any text editor
2. Add one email address per line
3. You can use partial matches - they will match anywhere in the sender's email
4. Lines starting with # are treated as comments

Example blacklist.txt:
```text
no-reply@
newsletter@
notifications@
marketing@
```
This would skip any emails from addresses containing these strings (e.g., no-reply@company.com, marketing@example.com)

## Q&A
- **Can I run this on X computer with a Y GPU?**
  - Non-M-chip Macs are not supported. Otherwise, as long as it comes with a M-Chip or a CUDA-compatible GPU, it should work.
- **Do I have enough GPU power to run this? Can I run smaller/quicker models?**
  - It reallydepends. In general, I'd recommend asking any LLMs for a proper setup of hyperparameters that you'd want to put in `config.py`. For Mac users, [mlx docs](https://github.com/ml-explore/mlx-examples/blob/main/lora/README.md) offers useful instructions (including how to convert the default LoRA model to QLoRA format).
- **Will this work with my Outlook email?**
  - No, this is a Gmail-only tool for now. I got some example scripts for Outlook, but they are not fully tested & integrated yet. The key difficulty is that there're many Gmail-specific APIs that have already been built into the current version.
- **Anything else?**
  - Feel free to file an issue on Github!

## TODO
- [x] CUDA QLoRA support
- [x] Windows support (now only tested on Mac)
- [x] Filter out spam ads emails by using replied rate (or is there some API for this?)
- [x] A configurable list of senders to ignore (e.g., no-reply@example.com)
- [ ] Shortcuts support for voice-input-based rewriting (whisper)
- [ ] RAG support for personal knowledge base
- [ ] Outlook Exchange support
- [ ] Labeling emails for priorities / categories
- [ ] ...

## License
MIT License - see LICENSE file for details

## Acknowledgements
- [MLX team](https://github.com/ml-explore/mlx) for bringing LLMs to M-chip Macs with increasingly improved APIs
- [LLMMe](https://github.com/pizzato/LLMMe) for initiating this project (I rewrote and streamlined the code to be more user-friendly)
