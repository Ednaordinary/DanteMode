import numpy as np
import numpy.lib.format


def np_save(file, array):
    magic_string = b"\x93NUMPY\x01\x00v\x00"
    header = bytes(("{'descr': '" + array.dtype.descr[0][1] + "', 'fortran_order': False, 'shape': " + str(
        array.shape) + ", }").ljust(127 - len(magic_string)) + "\n", 'utf-8')
    if type(file) == str:
        file = open(file, "wb")
    file.write(magic_string)
    file.write(header)
    file.write(array.data)


def np_load(file):
    if type(file) == str:
        file = open(file, "rb")
    header = file.read(128)
    if not header:
        return None
    descr = str(header[19:25], 'utf-8').replace("'", "").replace(" ", "")
    shape = tuple(int(num) for num in
                  str(header[60:120], 'utf-8').replace(',)', ')').replace(', }', '').replace('(', '').replace(')',
                                                                                                              '').split(
                      ','))
    datasize = numpy.lib.format.descr_to_dtype(descr).itemsize
    for dimension in shape:
        datasize *= dimension
    return np.ndarray(shape, dtype=descr, buffer=file.read(datasize))
