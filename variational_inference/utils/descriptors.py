import abc


class AutoStorage:
    """
    Stores and gets values from a given instance. On creation a unique identifier
    is created for the given value and the value is stored under the
    unique identifier in the instance.__dict__.
    """
    __counter = 0

    def __init__(self):
        cls = self.__class__
        prefix = cls.__name__
        index = cls.__counter
        self.storage_name = f"{prefix}#{index}"
        cls.__counter += 1

    def __get__(self, instance, owner):
        if instance is None:
            return self
        else:
            return getattr(instance, self.storage_name)

    def __set__(self, instance, value):
        if value is None:
            ValueError(f"You have to set a value.")
        else:
            setattr(instance, self.storage_name, value)


class Validated(AutoStorage, abc.ABC):
    """
    Serves as an Interface for descriptors. On validation attributes are
    set in the given instance.
    """
    def __set__(self, instance, value):
        value = self.validate(instance, value)
        super().__set__(instance, value)

    @abc.abstractmethod
    def validate(self, instance, value):
        """
        Returns a validated value or an error
        Args:
            instance:
            value:

        Returns:

        """


class NotNoneAttribute(Validated):
    """
    An discriptor for a classattribute, which can not be None and has no default value.
    """
    def validate(self, instance, value):
        if value is None:
            raise ValueError("value can not be None.")
        return value


class NonNegativeAttribute(Validated):
    """
    An discriptor for a class attribute, which can not be negative.
    """
    def validate(self, instance, value):
        if value < 0:
            raise ValueError("value can not be below zero")
        return value
