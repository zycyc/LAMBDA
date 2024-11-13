import os
import pickle
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from base64 import urlsafe_b64decode
import email
from bs4 import BeautifulSoup
import logging
import re


class GmailAPI:
    SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]

    def __init__(self):
        self.creds = self._get_credentials()
        self.service = build("gmail", "v1", credentials=self.creds)

    def _get_header(self, message: dict, header_name: str) -> str:
        """Extract header value from message"""
        headers = message["payload"]["headers"]
        header = next(
            (h for h in headers if h["name"].lower() == header_name.lower()), None
        )
        return header["value"] if header else ""

    def _get_credentials(self):
        creds = None
        if os.path.exists("token.pickle"):
            with open("token.pickle", "rb") as token:
                creds = pickle.load(token)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    "credentials.json", self.SCOPES
                )
                creds = flow.run_local_server(port=0)

            with open("token.pickle", "wb") as token:
                pickle.dump(creds, token)

        return creds

    def _clean_email_content(self, content: str) -> str:
        """Remove quoted text, signatures, and clean HTML content"""
        # First clean HTML if present
        if "<" in content and ">" in content:
            soup = BeautifulSoup(content, "html.parser")
            # Remove tracking images
            for img in soup.find_all("img"):
                img.decompose()
            # Get text content
            content = soup.get_text()

        # Split by common quote indicators
        lines = content.split("\n")
        cleaned_lines = []

        for line in lines:
            # Skip lines that typically indicate quoted content
            if any(
                line.strip().startswith(quote)
                for quote in [">", "|", "On ", "From: ", "Sent: ", "To: "]
            ):
                break
            # Skip empty lines at the start
            if not cleaned_lines and not line.strip():
                continue
            # If there's "On xxxx wrote" pattern in line, take everything before that
            wrote_pattern = re.compile(r"(?<=.)On [^\n]+wrote:", re.MULTILINE)
            if wrote_pattern.search(line):
                cleaned_lines.append(wrote_pattern.split(line)[0])
                break
            cleaned_lines.append(line)

        # Join lines and clean up extra whitespace
        cleaned_text = "\n".join(cleaned_lines).strip()
        # Remove multiple consecutive newlines
        cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)

        return cleaned_text

    def list_messages_with_label(self, label: str):
        """List all messages with specific label"""
        try:
            results = (
                self.service.users()
                .messages()
                .list(userId="me", labelIds=[label])
                .execute()
            )
            messages = results.get("messages", [])

            while "nextPageToken" in results:
                results = (
                    self.service.users()
                    .messages()
                    .list(
                        userId="me",
                        labelIds=[label],
                        pageToken=results["nextPageToken"],
                    )
                    .execute()
                )
                messages.extend(results.get("messages", []))

            return messages
        except Exception as e:
            logging.error(f"Error listing messages: {str(e)}")
            return []

    def get_message_detail(self, msg_id: str):
        """Get full message details"""
        try:
            return (
                self.service.users()
                .messages()
                .get(userId="me", id=msg_id, format="full")
                .execute()
            )
        except Exception as e:
            logging.error(f"Error getting message {msg_id}: {str(e)}")
            return None

    def get_thread_detail(self, thread_id: str):
        """Get all messages in a thread"""
        try:
            thread = (
                self.service.users()
                .threads()
                .get(userId="me", id=thread_id, format="full")
                .execute()
            )
            return thread.get("messages", [])
        except Exception as e:
            logging.error(f"Error getting thread {thread_id}: {str(e)}")
            return []

    def extract_conversation(
        self, thread_messages: list, max_len: int = 1000
    ) -> tuple[str, str]:
        """
        Extract conversation from thread messages and the latest clean message.
        Args:
            thread_messages: List of messages from get_thread_detail
            max_len: Maximum number of words to return
        Returns:
            Tuple of (conversation_history, latest_message)
        """
        # Get the conversation history from all but the last message
        conversation_parts = []
        word_count = 0

        # Process messages from oldest to newest (excluding the last one)
        for msg in thread_messages[:-1]:
            # Extract key information
            sender = self._get_header(msg, "From")
            date = self._get_header(msg, "Date")
            content = self._clean_email_content(self.get_message_text(msg))

            # Format this message
            message_text = f"On {date}, {sender} wrote:\n{content}\n\n"

            # Count words in this message
            message_words = len(message_text.split())

            # Check if adding this message would exceed max_len
            if word_count + message_words > max_len:
                # If this is the first message, take a truncated version of the latest info
                if not conversation_parts:
                    words = message_text.split()[-max_len:]
                    conversation_parts.append(" ".join(words))
                break

            conversation_parts.append(message_text)
            word_count += message_words

        # Get the latest message separately
        latest_message = ""
        if thread_messages:
            latest_msg = thread_messages[-1]
            latest_content = self.get_message_text(latest_msg)
            latest_message = self._clean_email_content(latest_content)

        return "".join(conversation_parts).strip(), latest_message.strip()

    def get_message_text(self, message: dict) -> str:
        """Extract text content from message"""
        if not message:
            return ""
        if "payload" not in message:
            return ""

        if "parts" in message["payload"]:
            return self._get_text_from_parts(message["payload"]["parts"])
        elif "body" in message["payload"]:
            return self._get_text_from_body(message["payload"]["body"])

        return ""

    def _get_text_from_parts(self, parts):
        html_content = []
        plain_content = []

        for part in parts:
            if part.get("mimeType") == "text/html":
                html = self._get_text_from_body(part["body"])
                if html:
                    html_content.append(BeautifulSoup(html, "html.parser").get_text())
            elif part.get("mimeType") == "text/plain":
                plain_content.append(self._get_text_from_body(part["body"]))
            elif "parts" in part:
                nested_content = self._get_text_from_parts(part["parts"])
                if nested_content:
                    html_content.append(nested_content)

        # Return HTML content if available, otherwise plain text
        if html_content:
            return "\n".join(filter(None, html_content))
        return "\n".join(filter(None, plain_content))

    def _get_text_from_body(self, body):
        if "data" in body:
            return urlsafe_b64decode(body["data"].encode("UTF-8")).decode("utf-8")
        return ""

    def create_draft(
        self, to: str, subject: str, message_text: str, thread_id: str = None
    ):
        """Create a draft email"""
        try:
            message = {
                "raw": self._create_message(to, subject, message_text),
                "threadId": thread_id,
            }

            draft = (
                self.service.users()
                .drafts()
                .create(userId="me", body={"message": message})
                .execute()
            )

            return draft
        except Exception as e:
            logging.error(f"Error creating draft: {str(e)}")
            return None

    def _create_message(self, to: str, subject: str, message_text: str) -> str:
        """Create email message"""
        from email.mime.text import MIMEText
        import base64

        message = MIMEText(message_text)
        message["to"] = to
        message["subject"] = subject

        return base64.urlsafe_b64encode(message.as_bytes()).decode()

    def check_thread_has_draft(self, thread_id: str) -> bool:
        """Check if a thread has any drafts"""
        try:
            # Get the thread details
            thread = (
                self.service.users().threads().get(userId="me", id=thread_id).execute()
            )

            # Check if any message in the thread has DRAFT label
            for message in thread.get("messages", []):
                if "DRAFT" in message.get("labelIds", []):
                    return True
            return False

        except Exception as e:
            logging.error(f"Error checking drafts for thread {thread_id}: {str(e)}")
            return False
