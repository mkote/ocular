from time import time

G_DO_TIME_LOGGING = True


class timed_block(object):
    def __init__(self, description=""):
        self.desc = description
        self.count = 0

    def __enter__(self):
        self.then = time()
        return self

    def __exit__(self, exp_type, exc_val, exc_tb):
        global G_DO_TIME_LOGGING
        if exp_type is None and G_DO_TIME_LOGGING and self.count == 0:
            now = time()
            print "\t\t%s took: %s" % (self.desc, now-self.then)
            self.count += 1
            return False


def timeit(f):

    def timed(*args, **kw):

        ts = time()
        result = f(*args, **kw)
        te = time()

        print 'func:%r args:[%r, %r] took: %2.4f sec' % (f.__name__, args, kw, te-ts)
        return result

    return timed
