import abc

# TODO: define classes for every distribution needed so far to store parameters and random data generation functionality


class Distribution(abc.ABC):

    @abc.abstractmethod
    def sample(self):
        pass
