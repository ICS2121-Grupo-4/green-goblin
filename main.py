#!/usr/local/bin python3
# -*- encoding: utf-8 -*-
import numpy as np
import re
import os
import sys
import pickle
import parser.main as parser
from sklearn import linear_model

# Without trailing slash
MODELS_FOLDER_PATH = "../models"


# Construye matriz de datos y la guarda serializada en archivo binario.
def build_matrix():
    # Parsear los datos del dataset.
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

    # Guarda objeto serializado a un archivo binario.
    with open("./A.matrix", "wb") as file:
        pickle.dump(A, file)

    print(np.shape(A))
    print(len(movies), len(ratings), len(users))


# Deprecada. Actualmente estamos usando módulo `sklearn`.
def lasso_function(x, tau, A, Y):
    return tau * np.linalg.norm(x, ord=1) + \
        (1/2) * np.linalg.norm(A.dot(x) - Y, ord=2)**2


# Lee archivo binario de la matriz de datos serializadas y la retorna.
def read_matrix():
    with open("./A.matrix", "rb") as matrix_file:
        A = pickle.load(matrix_file)
    return A


# Guarda un modelo serializado en un archivo binario.
def save_model(model_id, model):
    with open("{}/{}.model".format(MODELS_FOLDER_PATH, model_id), "wb") \
            as model_file:
        pickle.dump(model, model_file)


# Retorna el id más grande de los modelos serializados en archivos.
def get_last_fitted_model():
    try:
        return (
            max(map(
                lambda file: int(file.split(".")[0]),
                filter(
                    lambda p: re.match("^.*\.model$", p),
                    list(os.walk(MODELS_FOLDER_PATH))[0][2]
                )
            ))
        )
    # No existen modelos.
    except ValueError:
        return 0


# Función que construye los modelos
def fit(resume=True):
    # Obtiene el id del ultimo modelo calculado, a fin de empezar el cálculo
    # desde esa columna de la matriz y no desde cero.
    offset = get_last_fitted_model() + 1 if resume else 0
    # Lee y calcula la dimensión de la matriz de datos.
    A = read_matrix()
    shape = np.shape(A)
    # Para cada columna desde el offset en adelante, calcula el modelo.
    for col in range(offset, shape[1]):
        # A_prima: matriz A sin la columna de la película para la cual se está
        #          construyendo el modelo.
        # Y: columna de la película para la cual se está construyendo el
        #   modelo.
        A_prima = np.delete(A, col, axis=1)
        Y = A[:, col]
        # LassoVarsCV hace cross-validation (elección del tau) automáticamente.
        clf = linear_model.LassoLarsCV(n_jobs=1, max_iter=500)
        try:
            clf.fit(A_prima, Y)
            print(col, col/shape[1], clf.alpha)
            # Guarda el modelo obtenido como archivo. Si se quiere acceder a
            # las componentes del vector del modelo se debe acceder al
            # atributo `coef_` del objeto LassoVarsCV.
            save_model(col, clf)
        except ValueError:
            print("ValueError...")


def build_models_matrix():
    # TODO: Probably a better way to do this.
    num_models = get_last_fitted_model() + 1
    num_components = num_models - 1
    matrix = []
    for i in range(num_models):
        try:
            with open(MODELS_FOLDER_PATH + "/{}.model".format(i), "rb") as \
                    model_file:
                model = pickle.load(model_file)
                matrix.append(model.coef_)
        except FileNotFoundError:
            matrix.append(np.zeros(num_components))
    np_matrix = np.matrix(matrix)
    with open("./MODELS.matrix", "wb") as models_file:
        pickle.dump(np_matrix, models_file)


if __name__ == '__main__':
    actions = {
        "build": build_matrix,
        "fit": fit,
        "build_models_matrix": build_models_matrix
    }

    try:
        arg = sys.argv[1]
    except IndexError:
        raise Exception("No action was specified.")

    if arg in actions.keys():
        actions[arg]()
    else:
        raise Exception("No valid action was specified.")
