import logging
import logging.config
import logging.handlers
from path import Path


def init_logger(log_dir='./', filename='train.log', log_name='train_logger', level='INFO'):

    logger = logging.getLogger(log_name)
    fh = logging.handlers.RotatingFileHandler(Path(log_dir) / filename, 'w', 20 * 1024 * 1024, 5)
    formatter = logging.Formatter('%(asctime)s %(levelname)5s - %(name)s '
                                  '[%(filename)s line %(lineno)d] - %(message)s',
                                  datefmt='%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    fh = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s] %(message)s',)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.setLevel(level)
    logger.info("Start training")
    return logger
