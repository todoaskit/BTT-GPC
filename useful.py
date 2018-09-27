import time


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print('\t----- {}s'.format(int(te - ts)))
        return result
    return timed
