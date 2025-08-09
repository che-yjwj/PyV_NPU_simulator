
import logging
def get_logger(name:str="pyv-npu"):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    return logging.getLogger(name)
