import os
import sys
import logging
from pathlib import Path
import config

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup_user():
    """Setup user configuration if not already done."""
    # Check if USER_NAME is empty in config
    if config.USER_NAME:
        logger.info(f"User already configured: {config.USER_NAME}")
        return

    # Get user's name
    user_name = input(
        "Please enter your name (this will be used for telling the AI who you are): "
    ).strip()
    if not user_name:
        logger.error("Name cannot be empty.")
        sys.exit(1)

    # Update the USER_NAME in config
    try:
        config_path = Path("config.py")
        with open(config_path, "r") as f:
            content = f.read()

        # Replace the empty USER_NAME with actual name
        updated_content = content.replace(
            'USER_NAME = ""', f'USER_NAME = "{user_name}"'
        )

        with open(config_path, "w") as f:
            f.write(updated_content)

        logger.info(f"User configuration completed for: {user_name}")
    except Exception as e:
        logger.error(f"Failed to update config: {str(e)}")
        sys.exit(1)


def create_dataset():
    """Create the training dataset."""
    try:
        if not os.path.exists("credentials.json"):
            logger.error(
                "credentials.json not found. Please set up your Google credentials first."
            )
            sys.exit(1)

        logger.info("Creating training dataset...")
        os.system("python create_dataset.py --output email_dataset.csv")

        if not os.path.exists("email_dataset.csv"):
            raise FileNotFoundError("Dataset creation failed")

        logger.info("Dataset created successfully")
    except Exception as e:
        logger.error(f"Failed to create dataset: {str(e)}")
        sys.exit(1)


def train_model():
    """Train the model using the created dataset."""
    try:
        logger.info("Starting model training...")
        os.system(
            "python train_model.py --dataset email_dataset.csv --output model_output"
        )
        logger.info("Model training completed")
    except Exception as e:
        logger.error(f"Failed to train model: {str(e)}")
        sys.exit(1)


def run_email_bot():
    """Run the email bot."""
    try:
        logger.info("Starting email bot...")
        os.system("python lambda_bot.py")
    except Exception as e:
        logger.error(f"Failed to run email bot: {str(e)}")
        sys.exit(1)


def main():
    """Main function to orchestrate the workflow."""
    logger.info("Starting LAMBDA setup and execution...")

    # Check for required files
    required_files = [
        "create_dataset.py",
        "train_model.py",
        "lambda_bot.py",
        "config.py",
    ]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        logger.error(f"Missing required files: {', '.join(missing_files)}")
        sys.exit(1)

    # Execute workflow
    setup_user()

    # Ask user what they want to do
    while True:
        print("\nLAMBDA Options:")
        print("1. Create/Update training dataset")
        print("2. Train model")
        print("3. Run email bot")
        print("4. Run complete workflow (1-3)")
        print("5. Exit")

        choice = input("\nEnter your choice (1-5): ").strip()

        if choice == "1":
            create_dataset()
        elif choice == "2":
            train_model()
        elif choice == "3":
            run_email_bot()
        elif choice == "4":
            create_dataset()
            train_model()
            run_email_bot()
        elif choice == "5":
            logger.info("Exiting LAMBDA...")
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
