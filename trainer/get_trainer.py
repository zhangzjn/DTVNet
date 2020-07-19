from . import sky_timelapse_trainer

def get_trainer(name):
    if name == 'sky_timelapse':
        TrainFramework = sky_timelapse_trainer.TrainFramework
    else:
        raise NotImplementedError(name)

    return TrainFramework
