import ctypes as c
import numpy as np

class MyStruct(c.Structure):
    _fields_ = [("data", c.POINTER(c.c_double))]

# Scenario 1: Correct - memory owned by `arr`
arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
struct1 = MyStruct()
struct1.data = arr.ctypes.data_as(c.POINTER(c.c_double))
# struct1.data is valid as long as `arr` exists

print('#'* 50)
print(f"arr: {arr}")
print(f"struct1.data.contents: {struct1.data.contents}")
print('#'* 50)

# Scenario 2: Incorrect - `temp_arr` is temporary
struct2 = MyStruct()
temp_arr = np.array([4.0, 5.0, 6.0], dtype=np.float64)
struct2.data = temp_arr.ctypes.data_as(c.POINTER(c.c_double))
del temp_arr # temp_arr is garbage collected, struct2.data is now dangling
# Accessing struct2.data.contents here would likely cause a NULL pointer access or crash

print('#'* 50)
print(f"temp_arr: {temp_arr}")
print(f"struct2.data.contents: {struct2.data.contents}")
print('#'* 50)
