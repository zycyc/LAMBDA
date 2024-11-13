import os
import pandas as pd
from gmail_utils import GmailAPI
from typing import List, Dict
import logging
import argparse
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
import config


class EmailDatasetCreator:
    def __init__(self, gmail_api: GmailAPI):
        self.gmail_api = gmail_api

    def get_sent_emails(self) -> List[Dict]:
        """Get all sent emails and their context"""
        logging.info("Fetching sent emails...")

        # Get all sent emails
        sent_messages = self.gmail_api.list_messages_with_label("SENT")

        dataset = []
        for message in tqdm(sent_messages, desc="Processing emails"):
            try:
                # Get full message details
                msg_data = self.gmail_api.get_message_detail(message["id"])

                # Get original conversation and latest message
                original_thread = self.gmail_api.get_thread_detail(msg_data["threadId"])
                (original_conversation, latest_message) = (
                    self.gmail_api.extract_conversation(
                        original_thread, config.EMAIL_CONFIG["max_thread_length"]
                    )
                )

                # continue on forwarded emails
                if "Fwd:" in self._get_header(msg_data, "Subject"):
                    continue

                dataset.append(
                    {
                        "reply_from": self._get_header(msg_data, "From"),
                        "reply_to": self._get_header(msg_data, "To"),
                        "reply_cc": self._get_header(msg_data, "Cc"),
                        "reply_subject": self._get_header(msg_data, "Subject"),
                        "original_content": original_conversation,
                        "reply_content": latest_message,
                    }
                )

            except Exception as e:
                logging.warning(f"Error processing message {message['id']}: {str(e)}")
                continue

        # Convert dictionaries to tuples of items for deduplication
        unique_dataset = [dict(t) for t in {tuple(d.items()) for d in dataset}]

        return unique_dataset

    def _get_header(self, message: Dict, header_name: str) -> str:
        """Extract header value from message"""
        headers = message["payload"]["headers"]
        header = next(
            (h for h in headers if h["name"].lower() == header_name.lower()), None
        )
        return header["value"] if header else ""

    def create_dataset(self, output_file: str):
        """Create and save the dataset"""
        # Define a cache file name based on the output file
        cache_file = output_file.rsplit(".", 1)[0] + "_raw.csv"

        if os.path.exists(cache_file):
            logging.info(f"Loading cached raw data from {cache_file}")
            df = pd.read_csv(cache_file)
        else:
            dataset = self.get_sent_emails()
            if not dataset:  # Check if dataset is empty
                logging.error("No emails were processed. Dataset is empty.")
                return
            df = pd.DataFrame(dataset)
            # Save the raw data for future use
            df.to_csv(cache_file, index=False)
            logging.info(f"Raw data cached to {cache_file}")

        if not os.path.exists(output_file):
            # Verify DataFrame has required columns
            required_columns = [
                "reply_from",
                "reply_to",
                "reply_cc",
                "reply_subject",
                "original_content",
                "reply_content",
            ]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logging.error(f"Missing required columns: {missing_columns}")
                return

            df["prompt"] = df.apply(
                lambda x: f"""From: {x['reply_from']}
                To: {x['reply_to']}
                Cc: {x['reply_cc']}
                Subject: {x['reply_subject']}

                {x['original_content']}""",
                axis=1,
            )
            df["completion"] = df["reply_content"]

            # Save dataset
            df[["prompt", "completion"]].to_csv(output_file, index=False)
            logging.info(f"Dataset saved to {output_file} with {len(df)} examples")
        else:
            logging.error(f"Output file {output_file} already exists, skipping...")


def main():
    parser = argparse.ArgumentParser(
        description="Create email training dataset from Gmail"
    )
    parser.add_argument("--output", default="email_dataset.csv", help="Output CSV file")

    args = parser.parse_args()

    # Initialize Gmail API
    gmail_api = GmailAPI()

    # Create dataset
    creator = EmailDatasetCreator(gmail_api)
    creator.create_dataset(args.output)


if __name__ == "__main__":
    main()
