import logging
import sys

def get_logger(name: str) -> logging.Logger:
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    return logger

def interpolate(x, in_range, out_range):
    a, b = in_range      # e.g., [0, 1]
    c, d = out_range     # e.g., [10, 3]
    
    ratio = (x - a) / (b - a)
    return c + ratio * (d - c)

def stop_sign(str):
    str = str.lower()
    if "so the answer is" in str:
        return True
    elif "###" in str:
        return True
    elif "ans :" in str:
        return True
    else:
        return False