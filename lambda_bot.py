import logging
import re
import time
import os
from gmail_utils import GmailAPI
from typing import Optional, Tuple
import config
from glob import glob

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


class LambdaBot:
    def __init__(self, model_path: str, adapter_dir: Optional[str] = None):
        self.gmail_api = GmailAPI()
        self.model_path = model_path
        self.adapter_dir = adapter_dir or os.path.join(
            os.path.dirname(model_path), "adapters"
        )
        self.model, self.tokenizer = self._load_model()

    def _get_latest_adapter_path(self) -> Optional[str]:
        """Find the latest adapter file in the adapter directory"""
        if not os.path.exists(self.adapter_dir):
            return None

        # Look for all .safetensors files recursively through all subdirectories
        adapter_files = glob(
            os.path.join(self.adapter_dir, "**", "*.safetensors"), recursive=True
        )
        if not adapter_files:
            return None

        # Get the most recently created adapter file's folder path
        latest_adapter = max(adapter_files, key=os.path.getctime)
        latest_adapter_path = os.path.dirname(os.path.abspath(latest_adapter))
        logging.info(
            f"Found latest adapter: {latest_adapter} (Created: {time.ctime(os.path.getctime(latest_adapter))})"
        )
        return latest_adapter_path

    def _load_model(self) -> Tuple:
        """Load the appropriate model based on platform and available adapters"""

        if config.IS_MAC:
            from mlx_lm import load

            if self.adapter_dir:
                logging.info(f"Loading model with adapter from: {self.adapter_dir}")
                model, tokenizer = load(self.model_path, adapter_path=self.adapter_dir)
            else:
                logging.info("Loading base model without adapter")
                model, tokenizer = load(self.model_path)

        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            tokenizer = AutoTokenizer.from_pretrained(self.model_path)

            model = AutoModelForCausalLM.from_pretrained(
                self.model_path, torch_dtype=torch.float16, device_map="auto"
            )
            if self.adapter_dir:
                model.load_adapter(self._get_latest_adapter_path())

        return model, tokenizer

    def generate_response(self, email_thread: str) -> str:
        """Generate response using the loaded model"""
        # Create messages list for chat template
        messages = [
            {
                "role": "system",
                "content": str(
                    config.EMAIL_CONFIG["system_prompt"].format(
                        user_name=config.USER_NAME
                    )
                ),
            },
            {"role": "user", "content": str(email_thread)},
        ]

        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        if config.IS_MAC:
            from mlx_lm.utils import generate

            # MLX generation
            response = generate(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt=prompt,
                max_tokens=config.EMAIL_CONFIG["max_response_length"],
            )
            # logging.info(f"First generated response: {response}")
        else:  # Windows/Linux CUDA
            # Transformers generation
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=config.FINE_TUNE_CONFIG["max_seq_length"],
                truncation=True,
                padding=False,
            ).to("cuda")

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=config.EMAIL_CONFIG["max_response_length"],
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            prompt_length = inputs["input_ids"].shape[1]
            response = self.tokenizer.decode(
                outputs[0][prompt_length:], skip_special_tokens=True
            )

        # Let's ask the model to format our response a bit nicer
        format_prompt = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": config.EMAIL_CONFIG["format_prompt"]},
                {"role": "user", "content": response},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

        # Format response by generating again
        if config.IS_MAC:
            formatted_response = generate(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt=format_prompt,
                max_tokens=config.EMAIL_CONFIG["max_response_length"],
            )
            # logging.info(f"Formatted response: {formatted_response}")
        else:
            inputs = self.tokenizer(
                format_prompt,
                return_tensors="pt",
                max_length=config.FINE_TUNE_CONFIG["max_seq_length"],
                truncation=True,
                padding=False,
            ).to("cuda")
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=config.EMAIL_CONFIG["max_response_length"],
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            prompt_length = inputs["input_ids"].shape[1]
            formatted_response = self.tokenizer.decode(
                outputs[0][prompt_length:], skip_special_tokens=True
            )

        # Final manual cleanup
        # 1. if a line starts with "Subject: ", remove the line
        formatted_response = re.sub(
            r"^Subject:.*\n", "", formatted_response, flags=re.MULTILINE
        )
        # 2. if a line starts with "Re: ", remove the line
        formatted_response = re.sub(
            r"^Re:.*\n", "", formatted_response, flags=re.MULTILINE
        )
        # 3. if a line starts with "LAMBDA", remove the line
        formatted_response = re.sub(
            r"^LAMBDA.*\n", "", formatted_response, flags=re.MULTILINE
        )
        return config.RESPONSE_TEMPLATE.format(
            generated_response=formatted_response.strip(),
            bot_signature=config.EMAIL_CONFIG["bot_signature"],
        )

    def process_unread_emails(self):
        """Process all unread emails and create draft responses"""
        unread_messages = self.gmail_api.list_messages_with_label("UNREAD")

        # Load blacklist
        blacklist = set()
        blacklist_path = os.path.join(os.path.dirname(__file__), "blacklist.txt")
        if os.path.exists(blacklist_path):
            with open(blacklist_path, "r") as f:
                blacklist = {
                    line.strip().lower()
                    for line in f
                    if line.strip() and not line.startswith("#")
                }

        for message in unread_messages:
            try:
                # Check if already processed
                msg_detail = self.gmail_api.get_message_detail(message["id"])
                msg_labels = set(msg_detail.get("labelIds", []))

                # Get sender email
                sender = self.gmail_api._get_header(msg_detail, "From").lower()
                sender_email = (
                    sender[sender.find("<") + 1 : sender.find(">")]
                    if "<" in sender
                    else sender
                )

                # Skip if sender is in blacklist
                if any(blocked in sender_email for blocked in blacklist):
                    logging.info(
                        f"Skipping message {message['id']}, from: {sender} - sender is blacklisted"
                    )
                    continue

                # Skip if message has any of the ignored labels
                if msg_labels & config.EMAIL_CONFIG["ignore_labels"]:
                    logging.info(
                        f"Skipping message {message['id']}, snippet: {msg_detail['snippet']} - has ignored label(s): {msg_labels}"
                    )
                    continue

                # Skip if message has the bot label
                if config.EMAIL_CONFIG["bot_label"] in msg_labels:
                    continue

                # Get thread details
                thread_id = msg_detail["threadId"]

                # Skip if thread already has a draft
                if self.gmail_api.check_thread_has_draft(thread_id):
                    logging.info(f"Skipping thread {thread_id} - draft already exists")
                    continue

                thread_messages = self.gmail_api.get_thread_detail(thread_id)
                conversation_history, second_to_last_message, _ = (
                    self.gmail_api.extract_conversation(
                        thread_messages,
                        max_len=config.EMAIL_CONFIG["max_thread_length"],
                    )
                )

                # Prepare email context for prompt
                email_context = config.PROMPT_TEMPLATE.format(
                    reply_from=self.gmail_api._get_header(msg_detail, "From"),
                    reply_to=self.gmail_api._get_header(msg_detail, "To"),
                    reply_cc=self.gmail_api._get_header(msg_detail, "Cc"),
                    reply_subject=f"Re: {self.gmail_api._get_header(msg_detail, 'Subject')}",
                    context=conversation_history,
                    original_content=second_to_last_message,
                )

                # Generate response
                logging.info(
                    f"Generating response for email from: {self.gmail_api._get_header(msg_detail, 'From')}, snippet: {msg_detail['snippet']}"
                )
                response = self.generate_response(email_context)
                response = response.strip()

                # Create draft
                self.gmail_api.create_draft(
                    to=self.gmail_api._get_header(msg_detail, "From"),
                    subject=f"Re: {self.gmail_api._get_header(msg_detail, 'Subject')}",
                    message_text=response,
                    thread_id=msg_detail["threadId"],
                )

                logging.info(
                    f"Draft for email from: {self.gmail_api._get_header(msg_detail, 'From')} has been generated and saved \n"
                )

            except Exception as e:
                logging.error(f"Error processing message {message['id']}: {str(e)}")
                continue


def main():
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Run LAMBDA email bot")
    parser.add_argument("--adapter-dir", help="Directory containing adapter files")
    parser.add_argument(
        "--interval", type=int, default=300, help="Check interval in seconds"
    )

    args = parser.parse_args()

    # Check for default adapter directory
    default_adapter_dir = Path("model_output/adapters")
    adapter_dir = (
        args.adapter_dir
        if args.adapter_dir
        else (str(default_adapter_dir) if default_adapter_dir.exists() else None)
    )

    # if interval in config, use that, otherwise use args.interval
    if config.EMAIL_CONFIG["interval"]:
        args.interval = config.EMAIL_CONFIG["interval"]

    bot = LambdaBot(config.BASE_MODEL, adapter_dir)
    logging.info("Bot initialized! Now processing emails...")

    while True:
        try:
            logging.info("Processing unread emails...")
            bot.process_unread_emails()
            time.sleep(args.interval)
        except KeyboardInterrupt:
            logging.info("Bot stopped by user")
            break
        except Exception as e:
            logging.error(f"Error in main loop: {str(e)}")
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
