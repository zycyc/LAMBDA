# LAMBDA (LocalAutoMailBoxDraftAssistant)
A local AI-powered email automation system that learns from your email style and creates draft responses for every unread email in your (Gmail) inbox.

Motivations:
- Save time replying to emails -- **JUST Open the draft and Send**
- Keep all data and model to yourself -- **No cloud, no tracking**
- Something everyone can use everyday -- **Set up in less than 10 minutes and forget about it**

## Features
- Automatically extracts your email communication patterns from Gmail
- Fine-tunes LLaMA models on your email style using LoRA
- Creates AI-generated draft responses for unread emails
- Supports both Apple Silicon (MLX) and CUDA GPUs (NOTE: CUDA has NOT been implemented yet)

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

For Mac (Apple Silicon) and Linux:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For Linux:
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

3. Enable Gmail API and Setup Credentials (only takes about 5 minutes):
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

## TODO
- [ ] CUDA support
- [ ] Shortcuts support for voice-input-based rewriting (whisper)
- [ ] A configurable list of senders to ignore (e.g., no-reply@example.com)
- [ ] Windows support (now only tested on Mac)
- [ ] RAG support for personal knowledge base
- [ ] QLORA support
- [ ] Outlook Exchange support
- [ ] Labeling emails for priorities / categories
- [ ] Filter out spam ads emails by using replied rate (or is there some API for this?)
- [ ] ...

## License
MIT License - see LICENSE file for details

## Acknowledgements
- [MLX team](https://github.com/ml-explore/mlx) for bringing LLMs to M-chip Macs with increasingly improved APIs
- [LLMMe](https://github.com/pizzato/LLMMe) for initiating this project (I rewrote and streamlined the code to be more user-friendly)
