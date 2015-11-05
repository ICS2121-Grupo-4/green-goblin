__author__ = 'Carlos Tomi'
import functools
#with open('ml-100k/README', 'r') as file:
#    print(file.read())

#with open('ml-100k/u.item', 'r') as file:  ## movie id | movie title | release date | video release date |
#              #IMDb URL | unknown | Action | Adventure | Animation |
#              #Children's | Comedy | Crime | Documentary | Drama | Fantasy |
#              #Film-Noir | Horror | Musical | Mystery | Romance | Sci-Fi |
#              #Thriller | War | Western |
#              pass

#with open('ml-100k/u.genre', 'r') as file:  ## user id | age | gender | occupation | zip code
#    pass


class Evaluacion:

    def __init__(self, str):
        s = str.split('\t')
        self.user_id = s[0]
        self.item_id = s[1]
        self.rating = s[2]
        self.timestamp = s[3]


class User:
    id = 0
    def __init__(self, id):
        self.id = int(id)
        self.peliculas_evaluadas = 0
        if self.id > User.id:
            User.id = self.id


class Pelicula:
    id = 0
    def __init__(self, id):
        self.id = int(id)
        self.rating_total = 0
        self.evaluaciones = 0

    def rating_promedio(self):
        self.rating_promedio = self.rating_total/self.evaluaciones


evaluaciones = []

with open('ml-100k/u.data', 'r') as file:   ## user id | item id | rating | timestamp.
    lista = file.readlines()
    for i in range(len(lista)):
        evaluaciones.append(Evaluacion(lista[i]))

# usuarios = {}
# peliculas = {}
# for ev in evaluaciones:
#     if int(ev.user_id) >= User.id:
#         usuarios.update({int(ev.user_id): User(ev.user_id)})
#     if int(ev.user_id) >= User.id:
#         usuarios.update({int(ev.user_id): User(ev.user_id)})
#     usuarios[int(ev.user_id)].peliculas_evaluadas += 1
#     peliculas.update({ev.item_id: Pelicula(ev.item_id)})
#     peliculas[ev.item_id].rating_total += int(ev.rating)
#     peliculas[ev.item_id].evaluaciones += 1
#
# for p in peliculas:
#     p.rating_promedio()
#






