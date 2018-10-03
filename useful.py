import time


def timeit(method):
    def timed(*args, **kwargs):
        ts = time.time()
        result = method(*args, **kwargs)
        te = time.time()
        if 'print_result' not in kwargs or kwargs['print_result']:
            print('\t----- {}s'.format(int(te - ts)))
        return result
    return timed
