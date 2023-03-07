import logging

logger = logging.getLogger('interegular')
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.CRITICAL)


def per_char_repr(s: str):
    """
    joins together the repr of each char in the string, while throwing away the repr
    This for example turns `"'\""` into `'"` instead of `\'"` like the normal repr
    would do.
    """
    return ''.join(repr(c)[1:-1] for c in s)
