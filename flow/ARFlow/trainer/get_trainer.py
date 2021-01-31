from . import sintel_trainer, sintel_trainer_ar
from . import kitti_trainer, kitti_trainer_ar
from . import sky_trainer


def get_trainer(name):
    if name == 'Sintel':
        TrainFramework = sintel_trainer.TrainFramework
    elif name == 'Sintel_AR':
        TrainFramework = sintel_trainer_ar.TrainFramework
    elif name == 'KITTI':
        TrainFramework = kitti_trainer.TrainFramework
    elif name == 'KITTI_AR':
        TrainFramework = kitti_trainer_ar.TrainFramework
    elif name == 'sky':
        TrainFramework = sky_trainer.TrainFramework
    else:
        raise NotImplementedError(name)

    return TrainFramework
