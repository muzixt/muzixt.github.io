import sys
import re
import time


class ProgressBar(object):
    DEFAULT = "Progress: %(bar)s %(percent)3d%%"

    def __init__(self, mode, epoch=None, total_epoch=None, iters=None, current=None, width=30, symbol="\u2588",
                 output=sys.stdout):  # stdout
        assert len(symbol) == 1

        self.mode = mode
        self.iters = iters
        self.symbol = symbol
        self.output = output
        self.width = width
        self.current = current
        self.epoch = epoch
        self.total_epoch = total_epoch
        self.txt = ""
        self.args = {}
        self.pre_time = 0.0
        self.cur_time = 0.0
        self.total_time = 0.0
        self.started_time = 0.0

    def update(self, current, epoch=None, message: dict = {}):
        self.current = current
        if epoch:
            self.epoch = epoch
        txt = ""
        for k, v in message.items():
            txt += f'{k}:{v:.3f} '
        self.txt = txt

    def sec2time(self, sec):
        ''' Convert seconds to '#D days#, HH:MM:SS.FFF' '''
        # if hasattr(sec, '__len__'):
        #     return [self.sec2time(s) for s in sec]
        # xs = str(sec).split(".")[-1].ljust(3, '0')
        m, s = divmod(sec, 60)
        h, m = divmod(m, 60)
        # d, h = divmod(h, 24)
        pattern = r'%02d:%02d:%02d'

        return pattern % (h, m, s)

    def __call__(self):
        percent = self.current / float(self.iters)
        size = int(self.width * percent)
        bar = "\u2502" + self.symbol * size + " " * (self.width - size) + "\u2502"

        self.pre_time = time.time() if self.pre_time == 0 else self.pre_time

        self.cur_time = time.time()
        spend_time = round(self.cur_time - self.pre_time, 4)
        self.pre_time = self.cur_time

        self.started_time += spend_time
        self.total_time = self.started_time + spend_time * (self.iters - self.current)

        self.args.update({
            "mode": self.mode,
            "iters": self.iters,
            "bar": bar,
            "current": self.current,
            "percent": percent,
            "txt": self.txt,
            "epoch": self.epoch,
            "epochs": self.total_epoch,
            "time": spend_time,
            "started_time": self.sec2time(self.started_time),
            "total_time": self.sec2time(self.total_time)
        })
        message = "\033[1;31m{mode} Epoch: {epoch}/{epochs}\033[0m \033[1;33m {bar} \033[0m  \033[1;36m[ {txt} ]\033[0m \033[1;32m[ {current}/{iters} | {time} sec/it | {started_time}/{total_time} | {percent:.2%} ]\033[0m".format(
            **self.args)
        if self.current == self.iters:
            print("\r", message, file=self.output)
            self.started_time = 0.0
            self.total_time = 0.0
            self.pre_time = 0.0

            # print(self.current,self.total)
        else:
            print("\r" + message, file=self.output, end="")


if __name__ == "__main__":

    from time import sleep

    progress = ProgressBar("Train", total_epoch=10, iters=100, )
    for i in range(1, 10):
        for x in range(100):
            progress.update(x + 1, i, message={"mess": 1.00})
            progress()
            sleep(0.1)
        # progress.done()
