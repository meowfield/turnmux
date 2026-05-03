from __future__ import annotations

import logging
from pathlib import Path
import tempfile
import unittest

from turnmux.log_setup import configure_logging


class LogSetupTests(unittest.TestCase):
    def test_configure_logging_redacts_telegram_bot_token_urls(self) -> None:
        token = "1234567890:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghi"
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / "turnmux.log"
            logger = configure_logging(log_path)

            logging.getLogger("httpx").info(
                'HTTP Request: POST https://api.telegram.org/bot%s/getMe "HTTP/1.1 200 OK"',
                token,
            )
            for handler in logger.handlers:
                handler.flush()

            logged_text = log_path.read_text(encoding="utf-8")

        self.assertNotIn(token, logged_text)
        self.assertIn("https://api.telegram.org/bot<redacted>/getMe", logged_text)

    def test_configure_logging_redacts_telegram_file_download_token_urls(self) -> None:
        token = "1234567890:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghi"
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / "turnmux.log"
            logger = configure_logging(log_path)

            logging.getLogger("httpx").info(
                'HTTP Request: GET https://api.telegram.org/file/bot%s/photos/file_11.jpg "HTTP/1.1 200 OK"',
                token,
            )
            for handler in logger.handlers:
                handler.flush()

            logged_text = log_path.read_text(encoding="utf-8")

        self.assertNotIn(token, logged_text)
        self.assertIn("https://api.telegram.org/file/bot<redacted>/photos/file_11.jpg", logged_text)

    def test_configure_logging_redacts_url_encoded_telegram_file_download_token_urls(self) -> None:
        token = "1234567890:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghi"
        encoded_token = token.replace(":", "%3A")
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / "turnmux.log"
            logger = configure_logging(log_path)

            logging.getLogger("httpx").info(
                'HTTP Request: GET https://api.telegram.org/file/bot%s/photos/file_11.jpg "HTTP/1.1 200 OK"',
                encoded_token,
            )
            for handler in logger.handlers:
                handler.flush()

            logged_text = log_path.read_text(encoding="utf-8")

        self.assertNotIn(token, logged_text)
        self.assertNotIn(encoded_token, logged_text)
        self.assertIn("https://api.telegram.org/file/bot<redacted>/photos/file_11.jpg", logged_text)
