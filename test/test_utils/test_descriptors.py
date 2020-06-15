import pytest
from variational_inference.utils.descriptors import *


class TestDescriptors:
    """
    Test whether the descriptors assign values to a managed class or deny assignment on false creation.
    """

    @pytest.fixture(params=[
        (NonNegativeAttribute, -1),
        (NonNegativeAttribute, 1),
        (NotNoneAttribute, None),
        (NotNoneAttribute, 1)
    ])
    def descriptor_test_case(self, request):
        return request.param

# ================ Test Cases =================

    def test_descriptors(self, descriptor_test_case):
        descriptor, test_case = descriptor_test_case

        def generate_sample_class(descriptor, test_case):
            class SampleClass:
                managed_attribute = descriptor()

                def __init__(self, value):
                    self.managed_attribute = value

            return SampleClass(test_case)

        try:
            sample_class = generate_sample_class(descriptor, test_case)
            assert sample_class.managed_attribute == test_case
        except ValueError as e:
            assert e.__class__ is ValueError
