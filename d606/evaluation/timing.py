from time import time

G_DO_TIME_LOGGING = True


class timed_block(object):
    def __init__(self, description=""):
        self.desc = description

    def __enter__(self):
        self.then = time()
        return self

    def __exit__(self, exp_type, exc_val, exc_tb):
        global G_DO_TIME_LOGGING
        if exp_type is None and G_DO_TIME_LOGGING:
            now = time()
            print "\t\t%s took: %s" % (self.desc,
                                       now-self.then)
            return False
