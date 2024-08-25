import os

def write_data_in_py(file_name, kwarg):
        f = open(file_name, 'a')
        print(kwarg['data_name'] + ' =', kwarg['data'], file = f)
        f.close()
