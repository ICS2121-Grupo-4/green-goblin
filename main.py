#!/usr/local/bin python3
# -*- encoding: utf-8 -*-
import numpy as np
# import scipy.optimize
import os
import parser.main as parser
# import fom.main as fom

FILE_PATH = os.path.dirname(__file__)
MOVIES_FILE_PATH = os.path.join(
    FILE_PATH, "./movies-dataset/ml-1m/movies.dat")
RATINGS_FILE_PATH = os.path.join(
    FILE_PATH, "./movies-dataset/ml-1m/ratings.dat")
USERS_FILE_PATH = os.path.join(
    FILE_PATH, "./movies-dataset/ml-1m/users.dat")


def lasso_function(x, tau, A, Y):
    return tau * np.linalg.norm(x, ord=1) + \
        (1/2) * np.linalg.norm(A.dot(x) - Y, ord=2)**2

movies = parser.MoviesParser(MOVIES_FILE_PATH).parse()
ratings = parser.RatingsParser(RATINGS_FILE_PATH).parse()
users = parser.UsersParser(USERS_FILE_PATH).parse()

# Formar matriz A

print(len(movies), len(ratings), len(users))
