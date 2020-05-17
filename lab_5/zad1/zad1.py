import numpy
from matplotlib import pyplot as plt
from random import randint


def get_data(m, film_id, dir="./ml-latest-small/ratings.csv"):
    x = {}  # [id_uzytkownika][id_filmu]
    y = {}

    if film_id > m:
        row_width = m
    else:
        row_width = m-1

    with open(dir) as f:
        f.readline()
        for line in f:
            user_id, movie_id, rating, _ = line.split(",")
            if int(movie_id) == film_id:
                x[int(user_id)] = [0.0 for _ in range(row_width)]
                y[int(user_id)] = float(rating)

    with open(dir) as f:
        f.readline()
        for line in f:
            user_id, movie_id, rating, _ = line.split(",")

            if int(movie_id) <= m and int(user_id) in x:
                if int(movie_id) > film_id:
                    x[int(user_id)][int(movie_id) - 2] = float(rating)
                elif int(movie_id) < film_id:
                    x[int(user_id)][int(movie_id) - 1] = float(rating)

    user_id = list(x.keys())
    user_id.sort()
    x_matrix = []
    y_matrix = []
    for i in user_id:
        x_matrix.append(x[i] + [1.0])
        y_matrix.append(y[i])

    return x_matrix, y_matrix


def separate_data(x, y, validation_set_size):
    x_valid = []
    y_valid = []
    for _ in range(validation_set_size):
        index = randint(0, len(x) - 1)
        x_valid += [x.pop(index)]
        y_valid += [y.pop(index)]
    return x, y, x_valid, y_valid


def lin_reg(x_train, y_train, x_valid, y_valid):
    #
    # TRAINING
    #
    coef = numpy.array(numpy.linalg.lstsq(x_train, y_train, rcond=None)[0])
    pred = [numpy.sum(numpy.multiply(coef, numpy.array(row)))
            for row in x_train]
    y_axis = [i+1 for i in range(len(pred))]
    plt.title("TRAINING SET PREDICTIONS")
    plt.plot(y_axis, y_train, 'bo', markersize=5, label="valid rating")
    plt.plot(y_axis, pred, 'ro', markersize=2, label="prediction")
    plt.legend()
    plt.show()

    #
    # VALIDATION
    #
    pred = [numpy.sum(numpy.multiply(coef, numpy.array(row)))
            for row in x_valid]
    y_axis = [i+1 for i in range(len(pred))]
    plt.title("VALIDATION SET PREDICTIONS")
    plt.plot(y_axis, y_valid, 'bo', markersize=5, label="valid rating")
    plt.plot(y_axis, pred, 'ro', markersize=2, label="prediction")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    films = 1000
    film_id = 1
    validation_set_size = 15

    x, y = get_data(films, film_id)
    print("SET X SIZE:", len(x), "SET Y SIZE:", len(y))
    x_train, y_train, x_valid, y_valid = separate_data(x, y,
                                                       validation_set_size)
    lin_reg(x_train, y_train, x_valid, y_valid)
