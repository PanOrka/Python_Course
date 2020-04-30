from math import *
from random import randint
from statistics import mean


def pascal_triangle(n):
    for i in range(n):
        for k in range(i+1):
            print(comb(i, k), end=' ')
        print("")


def primes(n):
    bool_primes = [True for i in range(n)]

    for i in range(2, int(n**(1/2) + 1)):
        if bool_primes[i]:
            for k in range(2, n):
                if i*k >= n:
                    break
                bool_primes[i*k] = False
    
    return [i for i in range(2, n) if bool_primes[i]]


def throw_duplicates(list):
    no_dup = []
    for i in list:
        if i not in no_dup:
            no_dup += [i]
    return no_dup


def prime_factors(n):
    ret_list = []
    prime = 2
    while n!=1:
        alfa = 0
        while n%prime == 0:
            alfa += 1
            n //= prime
        if alfa > 0:
            ret_list += [(prime, alfa)]
        prime += 1
    return ret_list


def fraczero(n):
    if n == 0:
        return 0
    else:
        factorial = 1
        for i in range(2, n+1):
            factorial *= i
    
        zero = 0
        while factorial%10 == 0:
            zero += 1
            factorial //= 10
        return zero


def rome_arabic(num):
    num_dict = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    temp = num[-1]
    value = 0
    for n in num[ : :-1]:
        if num_dict[n] >= num_dict[temp]:
            value += num_dict[n]
            temp = n
        else:
            value -= num_dict[n]
        temp = n
    return value


def matcher(list): # dlugosc stala = 10
    l_dict = {0: "a", 5: "c", 6: "b"}
    ret = []
    for el in list:
        for l in l_dict:
            if el[l] != l_dict[l]:
                break
        else:
            ret += [el]
    return ret


def random_20():
    r_values = [randint(0, 20) for i in range(20)]
    print("Lista: ", r_values)
    print("Srednia: ", mean(r_values))
    max1 = max(r_values)
    min1 = min(r_values)
    print("Maksymalna: ", max1)
    print("Minimalna: ", min1)
    lista_bez_min = [i for i in r_values if i != min1]
    lista_bez_max = [i for i in r_values if i != max1]
    print("Maksymalna2: ", max(lista_bez_max))
    print("Minimalna2: ", min(lista_bez_min))
    lista_parz = [i for i in r_values if i%2 == 0]
    print("Ilosc parz: ", len(lista_parz))


def calculator():
    while (True):
        x = input("x = ")
        command = "from math import *\nprint(" + x + ")"
        try:
            exec(command)
        except(Exception):
            print("Something gone wrong, not supported operation")


def plot(x_axis = 160, y_axis = 80):
    x = input("Podaj funkcje f(x) (musi byc zalezna od x) = ")
    a = input("Podaj poczatek przedzialu: ")
    b = input("Podaj koniec przedzialu: ")

    plot = []
    for i in range(y_axis):
        line = []
        for k in range(x_axis):
            if k == int(x_axis/2):
                line += ["|"]
            elif i == (y_axis/2):
                line += ["-"]
            else:
                line += [" "]
        plot.append(line)

    try:
        start = eval(a)
        stop = eval(b)
    except(Exception):
        print("Bad input data")
        return
    
    # Zakladam ze uzytkownik wprowadzil dobre dane
    step = (stop - start)/(x_axis)
    values = []
    for i in range(x_axis-1):
        try:
            values += [eval(x.replace('x', "(" + str(start + i*step) + ")"))]
        except(Exception):
            values += [0]
    try:
        values += [eval(x.replace('x', "(" + str(stop) + ")"))]
    except(Exception):
        values += [0]

    max_value = max(values)
    min_value = min(values)

    if abs(max_value) > abs(min_value):
        lenght = abs(2*max_value)
        values2 = []
        for i in values:
            values2 += [round(abs((i - max_value)/lenght)*(y_axis - 1))] # ile % wzglednej odleglosci od maksymalnej
        for i in range(y_axis):
            for k in range(x_axis):
                if values2[k] == i:
                    plot[i][k] = '*'
    else:
        lenght = abs(2*min_value)
        values2 = []
        for i in values:
            values2 += [(y_axis - 1) - round(((i - min_value)/lenght)*(y_axis - 1))]
        for i in range(y_axis):
            for k in range(x_axis):
                if values2[k] == i:
                    plot[i][k] = '*'

    for i in plot:
        for k in i:
            print(k, end='')
        print("")
