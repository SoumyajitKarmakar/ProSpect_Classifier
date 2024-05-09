from datetime import datetime
from functools import lru_cache
import json
import os
import os.path as osp
import torch
from PIL import Image

cwd = os.getcwd()
LOG_DIR = os.getenv('LOG_DIR', 'data_log')

@lru_cache  # same datestr on different calls
def get_datetimestr():
    # only go to 3 ms digits
    return datetime.now().strftime("%Y.%m.%d_%H.%M.%S")


def get_formatstr(n):
    # get the format string that pads 0s to the left of a number, which is at most n
    digits = 0
    while n > 0:
        digits += 1
        n //= 10
    return f"{{:0{digits}d}}"
