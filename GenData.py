import numpy as np

def serialize_array(arr, filename):
    with open(filename, 'w') as f:
        for item in arr:
            # write each item with precision of double (default in python)
            f.write("%.16f " % item)

# Test the function
arr = np.array([1.1234567891234567, 2.2345678912345678, 3.3456789123456789, 4.4567891234567890, 5.5678912345678901], dtype=np.float64)
serialize_array(arr, 'array_data.txt')
