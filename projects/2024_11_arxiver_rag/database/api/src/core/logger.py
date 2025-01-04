import logging


handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - [%(pathname)s:%(lineno)d - %(funcName)s] - %(message)s")
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)