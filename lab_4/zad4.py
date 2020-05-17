from inspect import getfullargspec
from math import sqrt


class fn:
    def __init__(self):
        self.functions = {}


    def add_func(self, f, args_n):
        self.functions[args_n] = f


    def __call__(self, *args):
        f = self.functions[len(args)]
        return f(*args)
fn_inst = fn()


def overload(f):
    fn_inst.add_func(f, len(getfullargspec(f).args))
    return fn_inst


@overload
def norm(x, y):
    return sqrt(x*x + y*y)


@overload
def norm(x, y, z):
    return abs(x) + abs(y) + abs(z)


if __name__ == "__main__":
    print("1 =", norm(2, 4))
    print("2 =", norm(2, 3, 4))