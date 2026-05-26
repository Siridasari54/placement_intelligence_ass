import time


class LatencyEval:

    def measure(self, function, *args, **kwargs):

        start = time.time()

        result = function(*args, **kwargs)

        end = time.time()

        return {
            "latency": end - start,
            "result": result
        }