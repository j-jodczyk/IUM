import numpy as np
import logging
import os

# defining log file:
def config_logging(name: int) -> None:
    # create a directory for logging
    log_dir = "log"
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    logging.basicConfig(
        level=logging.DEBUG,
        format="{asctime} {levelname:<8} {message}",
        style="{",
        filename=f"./log/{name}.log" ,
        filemode="a",
        force=True,
    )
    

def config_seed(seed:int=42):
    np.random.seed(seed)


def config(log_name:str="rest", seed:int=42):
    config_seed()
    config_logging(log_name)