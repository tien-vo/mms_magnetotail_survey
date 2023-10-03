import sys
import logging
import pint_xarray
from os import environ

_fmt = (
    r"%(asctime)s [%(levelname)s] "
    r"[%(filename)s.%(funcName)s(%(lineno)d)]: %(message)s"
)
logging.basicConfig(
    format=_fmt,
    level=logging.INFO if environ.get("DEBUG") is None else logging.DEBUG,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logging.captureWarnings(True)
