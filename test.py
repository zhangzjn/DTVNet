import json
import pprint
import datetime
import argparse
# from path import Path
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
    cfg.train.isTrain = False
    cfg.train.load_epoch = 200 # 160 for test and 170 for train

    # init logger
    cfg.save_root = Path(cfg.train.checkpoints) / cfg.train.name / '200708162546'
    logger = init_logger(log_dir=cfg.save_root, filename='test.log', log_name='test_logger')
    logger.info('=> testing: will save everything to {}'.format(cfg.save_root))

    # show configurations
    cfg_str = pprint.pformat(cfg)
    logger.info('=> configurations \n ' + cfg_str)

    # create datasets
    train_loader, valid_loader = get_dataset(cfg)

    # train
    TrainFramework = get_trainer(cfg.trainer)
    trainer = TrainFramework(cfg, train_loader, valid_loader, logger)
    trainer.test()
