import time


def check_time(func):
    def new_func(*args, **kwargs):
        t_start = time.process_time()
        func(*args, **kwargs)
        print("Function execution time:", time.process_time() - t_start)
    return new_func


@check_time
def ex_func(a, b):
    return a+b


if __name__ == "__main__":
    ex_func(2, 4)