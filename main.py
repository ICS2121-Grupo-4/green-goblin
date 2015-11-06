#!/usr/local/bin python3
# -*- encoding: utf-8 -*-
import numpy as np
# import scipy.optimize
import os
import sys
import pickle
import parser.main as parser
# import fom.main as fom


def build_matrix():
    FILE_PATH = os.path.dirname(__file__)
    MOVIES_FILE_PATH = os.path.join(
        FILE_PATH, "./movies-dataset/ml-1m/movies.dat")
    RATINGS_FILE_PATH = os.path.join(
        FILE_PATH, "./movies-dataset/ml-1m/ratings.dat")
    USERS_FILE_PATH = os.path.join(
        FILE_PATH, "./movies-dataset/ml-1m/users.dat")

    movies = parser.MoviesParser(MOVIES_FILE_PATH).parse()
    ratings = parser.RatingsParser(RATINGS_FILE_PATH).parse()
    users = parser.UsersParser(USERS_FILE_PATH).parse()

    n_movies = max(list(map(
        lambda movie: movie.movie_id,
        movies
    )))

    # Construir matriz A
    A = []

    # Cada usuario es una fila, cada película una columna
    for user in users:
        user_ratings = list(filter(
            lambda rating: rating.user_id == user.user_id,
            ratings
        ))
        row = np.zeros(n_movies)
        for rating in user_ratings:
            row[rating.movie_id - 1] = rating.rating
        A.append(row)

    A = np.array(A)

    with open("./A.matrix", "wb") as file:
        pickle.dump(A, file)

    print(np.shape(A))
    print(len(movies), len(ratings), len(users))


def lasso_function(x, tau, A, Y):
    return tau * np.linalg.norm(x, ord=1) + \
        (1/2) * np.linalg.norm(A.dot(x) - Y, ord=2)**2


def read_matrix():
    with open("./A.matrix", "rb") as matrix_file:
        A = pickle.load(matrix_file)
    print(np.shape(A))

if __name__ == '__main__':
    actions = {
        "build": build_matrix,
        "read": read_matrix
    }

    try:
        arg = sys.argv[1]
    except IndexError:
        raise Exception("No action was specified.")

    if arg in actions.keys():
        actions[arg]()
    else:
        raise Exception("No valid action was specified.")
