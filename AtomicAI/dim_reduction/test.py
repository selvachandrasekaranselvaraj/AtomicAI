import time
from multiprocessing import Pool


def square(x):
    print(f"start process:{x}")
    square = x * x
    print(f"square {x}:{square}")
    time.sleep(1)
    print(f"end process:{x}")


if __name__ == "__main__":
    starttime = time.time()
    pool = Pool()
    pool.map(square, range(0, 5))
    pool.close()
    endtime = time.time()
    print(f"Time taken {endtime-starttime} seconds")
