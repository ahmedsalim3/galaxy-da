import logging
import os
from datetime import datetime


# pylint: disable=broad-except
class Logger:
    """
    A Logger class for logging messages with a specific log level.

    The class follows the singleton design pattern, ensuring that only one
    instance of the Logger is created. The parameters of the first instance
    are preserved across all instances.
    """

    __instance = None

    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = super(Logger, cls).__new__(cls)
        return cls.__instance

    def __init__(self, log_file: str = None):
        if not hasattr(self, "_singleton_initialized"):
            self._singleton_initialized = True  # to prevent multiple initializations
            log_level = os.environ.get("LOG_LEVEL", str(logging.INFO))
            try:
                self.log_level = int(log_level)
            except Exception as err:
                self.dump_log(
                    f"Exception while parsing $LOG_LEVEL."
                    f"Expected int but it is {log_level} ({str(err)})."
                    "Setting app log level to info."
                )
                self.log_level = logging.INFO

            self.log_file = os.environ.get("LOG_FILE", log_file)
        else:
            if log_file is not None:
                self.log_file = log_file

    def info(self, message):
        """
        Set log level to 20 to see info messages
        export LOG_LEVEL=20
        """
        if self.log_level <= logging.INFO:
            self.dump_log(f"{message}")

    def debug(self, message):
        """
        Set log level to 10 to see debug messages
        export LOG_LEVEL=10
        """
        if self.log_level <= logging.DEBUG:
            self.dump_log(f"🕷️ {message}")

    def warning(self, message):
        """
        Set log level to 30 to see warning messages
        export LOG_LEVEL=30
        """
        if self.log_level <= logging.WARNING:
            self.dump_log(f"⚠️ {message}")

    def error(self, message):
        """
        Set log level to 40 to see error messages
        export LOG_LEVEL=40
        """
        if self.log_level <= logging.ERROR:
            self.dump_log(f"🔴 {message}")

    def critical(self, message):
        """
        Set log level to 50 to see critical messages
        export LOG_LEVEL=50
        """
        if self.log_level <= logging.CRITICAL:
            self.dump_log(f"💥 {message}")

    def set_log_file(self, log_file: str):
        self.log_file = log_file

    def dump_log(self, message):
        timestamp = str(datetime.now())[2:-7]
        formatted = f"{timestamp} - {message}"
        print(formatted)

        if self.log_file:
            try:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(formatted + "\n")
            except Exception as e:
                print(f"Failed to write log to {self.log_file}: {e}")


def log_config(data: dict, title: str):
    logger = Logger()

    if not data:
        return
    key_w = max(len(str(k)) for k in data)
    val_w = max(len(str(v)) if v is not None else 4 for v in data.values())
    val_w = max(val_w, 12)

    def _line(left, mid, right, fill="─"):
        return left + fill * (key_w + 2) + mid + fill * (val_w + 2) + right

    t_w = (key_w + 2) + (val_w + 2) + 3

    # title
    logger.info(
        "┌"
        + "─" * ((t_w - len(title) - 4) // 2)
        + f" {title} "
        + "─" * ((t_w - len(title) - 3) // 2)
        + "┐"
    )

    # header
    logger.info(f"│ {'Key'.ljust(key_w)} │ {'Value'.ljust(val_w)} │")
    logger.info(_line("├", "┼", "┤"))

    # rows
    for k, v in data.items():
        k, v = str(k), "None" if v is None else str(v)
        logger.info(f"│ {k.ljust(key_w)} │ {v.ljust(val_w)} │")

    logger.info(_line("└", "┴", "┘"))
