from . import patch
from .data.constant import *

import sys
import logging

logger = logging.getLogger("")
logger.setLevel(logging.INFO)
format = logging.Formatter("%(asctime)-10s %(message)s", "%H:%M:%S")

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(format)
logger.addHandler(handler)

__version__ = "0.2.0"