"""\
Code to timeout with processes.


References
----------
Code in this module comes from an activestate code recipe [1]_. For an
asynchronous solution see the active activestate code recipe [2]_.

.. _synchronous: http://code.activestate.com/recipes/577853-timeout-decorator-with-multiprocessing/

.. _asynchronous: http://code.activestate.com/recipes/577028/

.. [1] timeout decorator (with multiprocessing) (Python recipe) synchronous_
.. [2] Timeout Any Function (Python recipe) asynchronous_

Examples
--------
>>> timed_longcos = timeout(2)(_longcos)
>>> timed_longcos(1, 0)
0.5403...
>>> timed_longcos(1, 2) # doctest: +IGNORE_EXCEPTION_DETAIL
Traceback (most recent call last):
...
TimeoutException: timed out after 2 seconds

Notes
-----
The following examples from the original activestate code recipe
demonstrate how to use timeout as a decorator.  They don't seem to work for
me as the functions must be defined in __main__ to be pickled.  you get the
idea though.



.. code-block:: python

    @timeout(.5)
    def sleep(x):
        print "ABOUT TO SLEEP {0} SECONDS".format(x)
        time.sleep(x)
        return x

    sleep(1)
    Traceback (most recent call last):
       ...
    TimeoutException: timed out after 0 seconds

    sleep(.2)
    0.2

    @timeout(.5)
    def exc():
        raise Exception('Houston we have problems!')

    exc()
    Traceback (most recent call last):
       ...
    Exception: Houston we have problems!


"""
#Someuseful stuff
#http://stackoverflow.com/questions/492519/timeout-on-a-python-function-call/14924210#14924210
#http://stackoverflow.com/a/14924210/2530083
#http://stackoverflow.com/questions/7194884/assigning-return-value-of-function-to-a-variable-with-multiprocessing-and-a-pr
#http://eli.thegreenplace.net/2012/01/16/python-parallelizing-cpu-bound-tasks-with-multiprocessing/
#http://code.activestate.com/recipes/577853-timeout-decorator-with-multiprocessing/



import multiprocessing
import time
import logging
#logger = multiprocessing.log_to_stderr()
#logger.setLevel(logging.INFO)


class TimeoutException(Exception):
    pass


class RunableProcessing(multiprocessing.Process):
    def __init__(self, func, *args, **kwargs):
        self.queue = multiprocessing.Queue(maxsize=1)
        args = (func,) + args
        multiprocessing.Process.__init__(self, target=self.run_func, args=args, kwargs=kwargs)

    def run_func(self, func, *args, **kwargs):
        try:
            result = func(*args, **kwargs)
            self.queue.put((True, result))
        except Exception as e:
            self.queue.put((False, e))

    def done(self):
        return self.queue.full()

    def result(self):
        return self.queue.get()


def timeout(seconds, force_kill=True):
    def wrapper(function):
        def inner(*args, **kwargs):
            now = time.time()
            proc = RunableProcessing(function, *args, **kwargs)
            proc.start()
            proc.join(seconds)
            if proc.is_alive():
                if force_kill:
                    proc.terminate()
                runtime = int(time.time() - now)
                raise TimeoutException('timed out after {0} seconds'.format(runtime))
            assert proc.done()
            success, result = proc.result()
            if success:
                return result
            else:
                raise result
        return inner
    return wrapper



def _longcos(x, wait=0):
    """calc cos(x) after waiting `wait` seconds. max wait is 5 seconds"""

    import math
    wait=min(wait,5)
    time.sleep(wait)

    return math.cos(x)

if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['nose', '--verbosity=3', '--with-doctest', '--doctest-options=+ELLIPSIS'])
#    nose.runmodule(argv=['nose', '--verbosity=3'])

