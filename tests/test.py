print("just a test")

# test nd_array

import numpy as np
from numpy import ndarray

def test_ndarray():
    # Create a sample ndarray
    arr = np.array([[1, 2, 3], [4, 5, 6]])

    # Check if the type is ndarray
    assert isinstance(arr, ndarray), "The object is not an ndarray"
    print("Test passed: The object is an ndarray")

    print(type(arr))
    print((type(arr[0])))

test_ndarray()