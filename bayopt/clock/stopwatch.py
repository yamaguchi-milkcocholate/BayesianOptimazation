import time


class StopWatch(object):

    def __init__(self):
        self.start = time.time()

    def passed_time(self):
        return time.time() - self.start

    def reset(self):
        self.start = time.time()
