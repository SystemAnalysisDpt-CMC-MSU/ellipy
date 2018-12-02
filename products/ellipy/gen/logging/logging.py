import logging
from ellipy.gen.common.common import get_caller_name


def get_logger():
    caller_name = get_caller_name(2, 'full')
    logger = logging.getLogger(caller_name)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s [%(name)-12s] %(levelname)-8s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger
