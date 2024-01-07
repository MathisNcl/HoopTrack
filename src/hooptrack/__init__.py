"""Low-level init package"""

import logging

VERSION = (0, 0, 1)
__version__ = ".".join(map(str, VERSION))

logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

logger.addHandler(console_handler)
