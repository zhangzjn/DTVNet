import os


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, i=1, precision=3, names=None):
        self.meters = i
        self.precision = precision
        self.reset(self.meters)
        if names is not None:
            assert self.meters == len(names)
            self.names = names
        else:
            self.names = [''] * self.meters

    def reset(self, i):
        self.val = [0] * i
        self.avg = [0] * i
        self.sum = [0] * i
        self.count = [0] * i

    def update(self, val, n=1):
        if not isinstance(val, list):
            val = [val]
        if not isinstance(n, list):
            n = [n] * self.meters
        assert (len(val) == self.meters and len(n) == self.meters)
        for i in range(self.meters):
            self.count[i] += n[i]
            self.val[i] = val[i]
            self.sum[i] += val[i] * n[i]
            self.avg[i] = self.sum[i] / self.count[i]

    def __repr__(self):
        val = ' '.join(['{} {:.{}f}'.format(n, v, self.precision) for n, v in
                        zip(self.names, self.val)])
        avg = ' '.join(['{} {:.{}f}'.format(n, a, self.precision) for n, a in
                        zip(self.names, self.avg)])
        return '{} ({})'.format(val, avg)


def mkdirs(paths):
    if isinstance(paths, list):
        for path in paths:
            mkdir(path)
    elif isinstance(paths, str):
        mkdir(paths)
    else:
        raise NameError('[{} is neither a list or a str]'.format(paths))


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
