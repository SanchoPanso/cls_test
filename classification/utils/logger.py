import os
import sys
import logging


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    
    color = "\x1b[34;20m"
    reset = "\x1b[0m"
    formatter = logging.Formatter(f"{color}%(name)s:{reset} %(message)s")
    # formatter = logging.Formatter("%(name)s %(asctime)s %(levelname)s %(message)s")

    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger


# class CustomFormatter(logging.Formatter):

#     grey = "\x1b[38;20m"
#     blue = "\x1b[34;20m"
#     yellow = "\x1b[33;20m"
#     red = "\x1b[31;20m"
#     bold_red = "\x1b[31;1m"
#     reset = "\x1b[0m"
#     #format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

#     COLORS = {
#         logging.DEBUG: grey + format + reset,
#         logging.INFO: blue + format + reset,
#         logging.WARNING: yellow + format + reset,
#         logging.ERROR: red + format + reset,
#         logging.CRITICAL: bold_red + format + reset
#     }

#     def format(self, record):
#         color = self.COLORS.get(record.levelno)
#         log_fmt = f"{color}%(name)s:{self.reset} %(message)s"
#         formatter = logging.Formatter(log_fmt)
#         return formatter.format(record)


