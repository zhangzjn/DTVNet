import json
import pprint
import datetime
import argparse
from path import Path
from easydict import EasyDict

from utils.logger import init_logger
from utils.torch_utils import init_seed
from datasets.get_dataset import get_dataset
from trainer.get_trainer import get_trainer

import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    # init parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='configs/sky_timelapse.json')
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = EasyDict(json.load(f))
    # init seed
    init_seed(cfg.train.seed)

    # init logger
    if cfg.save_root == '':
        curr_time = datetime.datetime.now().strftime("%y%m%d%H%M%S")
        cfg.save_root = Path(cfg.train.checkpoints) / cfg.train.name / curr_time
        cfg.save_root.makedirs_p()
    logger = init_logger(log_dir=cfg.save_root)
    logger.info('=> training: will save everything to {}'.format(cfg.save_root))

    # show configurations
    cfg_str = pprint.pformat(cfg)
    logger.info('=> configurations \n ' + cfg_str)

    # create datasets
    train_loader, valid_loader = get_dataset(cfg)

    # train
    TrainFramework = get_trainer(cfg.trainer)
    trainer = TrainFramework(cfg, train_loader, valid_loader, logger)
    trainer.train()
