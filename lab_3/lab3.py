from functools import reduce


def transposition(matrix):
    return [" ".join(row.split()[x] for row in matrix) for x in range(len(matrix))]


def flatten(lst):
    for el in lst:
        if isinstance(el, list):
            for sub_el in flatten(el):
                yield sub_el
        else:
            yield el


def count_size(file):
    try:
        with open(file) as f:
            val = [int(line.split()[-1]) for line in f.readlines()]
            return sum(val)

    except FileNotFoundError:
        print("Nie udało się otworzyć pliku: " + file + "\n")


def quick_sort(lst):
    if len(lst) > 1:
        x = lst.pop()
        left = list(filter(lambda y: y < x, lst))
        right = list(filter(lambda y: y >= x, lst))
        return quick_sort(left) + [x] + quick_sort(right)
    else:
        return lst


def get_powerset(gset):
    return reduce(lambda res, x: res + [subset + [x] for subset in res], gset, [[]])

