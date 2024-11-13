class FeatureExtractor(object):

    def __init__(self, gen_function):
        self.gen_function = gen_function
    
    def __call__(self, *args, **kwds):
        return self.gen_function(*args, **kwds)
