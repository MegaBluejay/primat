class CountingFunc:
    def __init__(self, f):
        self.f = f
        self.n = 0

    def __call__(self, *args, **kwargs):
        self.n += 1
        return self.f(*args, **kwargs)
