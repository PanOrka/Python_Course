import numpy


def get_data(max_size, dir="/home/pan_orka/Code/Python/kurs_python/lab5/"
                           "zad1/ml-latest-small/ratings.csv"):
    movie_ratings = {}  # [user_id][movie_id]
    with open(dir) as f:
        f.readline()
        for line in f:
            user_id, movie_id, rating, _ = line.split(",")
            if int(movie_id) <= max_size:
                if int(user_id) in movie_ratings:
                    movie_ratings[int(user_id)][int(movie_id) - 1] = (
                        float(rating))
                else:
                    movie_ratings[int(user_id)] = numpy.zeros((max_size, ),
                                                              dtype=float)
                    movie_ratings[int(user_id)][int(movie_id) - 1] = (
                        float(rating))

    print("Loaded users:", len(movie_ratings),
          "| ID films from 1 to", max_size)
    movie_ratings = numpy.array([movie_ratings[x] for x in movie_ratings])

    return movie_ratings


def find_similar(movie_ratings, my_ratings, list_size, dir="/home/pan_orka/"
                 "Code/Python/kurs_python/lab5/zad1/ml-latest-small/"
                 "movies.csv"):
    numpy.seterr(divide='ignore', invalid='ignore')

    x = numpy.nan_to_num(movie_ratings /
                         numpy.linalg.norm(movie_ratings, axis=0))
    z = numpy.dot(x, numpy.nan_to_num(my_ratings /
                                      numpy.linalg.norm(my_ratings, axis=0)))
    z = numpy.nan_to_num(z/numpy.linalg.norm(z))
    answer = numpy.dot(x.T, z).flatten()

    answer_with_titles = []
    with open(dir) as f:
        f.readline()
        for line in f:
            splited = line.split(",")
            movie_id = int(splited[0])
            if movie_id > answer.shape[0]:
                break
            else:
                title = "".join(x for x in splited[1:-1])
                answer_with_titles += [[answer[movie_id - 1], movie_id, title]]

    answer_with_titles = sorted(
        answer_with_titles, key=lambda x: x[0], reverse=True)

    print("\ncos(theta), movie_id, movie_title")
    for a in answer_with_titles[:list_size]:
        print(a)


if __name__ == "__main__":
    max_size = 10000
    list_size = 40
    movie_ratings = get_data(max_size)

    my_ratings = numpy.zeros((max_size, 1))  # id shift, bo najmniejsze id to 1
    my_ratings[2571-1] = 5
    my_ratings[32-1] = 4
    my_ratings[260-1] = 5
    my_ratings[1097-1] = 4

    find_similar(movie_ratings, my_ratings, list_size)
