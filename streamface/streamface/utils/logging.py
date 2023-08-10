import datetime


class Logging(object):
    def __init__(self, filepath):
        self.logger = open(filepath, 'w')

    def log(self, msg, msg_type, screen=True):
        msg = '{}\t{}\t{}\n'.format(self.timestamp(), msg_type, msg)
        self.logger.write(msg)
        if screen:
            print(msg, end='', flush=True)

    def timestamp(self):
        ts = datetime.datetime.now()
        return ts

    def flush(self):
        self.logger.flush()

    def close(self):
        self.logger.close()
