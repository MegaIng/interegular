import logging

logger = logging.getLogger('interegular')
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.CRITICAL)
