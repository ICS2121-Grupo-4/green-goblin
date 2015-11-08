#!/usr/local/bin python3
# -*- encoding: utf-8 -*-
import numpy as np


class FOM:
    def __init__(self, function, x0, args=(), step=0.1, stop=0.001):
        print("Constructing FOM instance...")
        self.step = step
        self.function = lambda x: function(x, *args)
        self.stop = stop
        self.x0 = x0

    def optimize(self):
        x = self.x0
        print("  i ;   f(x)  ;   |x|   ;  grad   ; delta")
        i = 1
        while True:
            # Calcular un subgradiente
            gradient = self.numerical_gradient(x)

            # Calcular nuevo punto
            new_x = x - gradient * self.step

            # Delta
            delta = abs(self.function(new_x) - self.function(x))

            print("{: >4d}; {: <8.5f}; {: <8.5f}; {: <8.5f}; {: <8.5f}".format(
                i,
                self.function(x),
                np.linalg.norm(x),
                np.linalg.norm(gradient),
                delta
            ))

            # Criterio de parada es distancia entre puntos
            if delta <= self.stop:
                return new_x
            else:
                x = new_x
                i += 1

    def numerical_gradient(self, x, gradient_epsilon=0.00001):
        grad = np.zeros(len(x))
        for i in range(len(x)):
            print(i/len(x))
            temp_x = [np.copy(x), np.copy(x)]
            temp_x[0][i] = temp_x[0][i] + gradient_epsilon
            temp_x[1][i] = temp_x[1][i] - gradient_epsilon
            grad[i] = (self.function(temp_x[0]) - self.function(temp_x[1])) \
                / (2 * gradient_epsilon)
        norm = np.linalg.norm(grad)
        return grad / np.linalg.norm(grad) if norm > 0 else grad

if __name__ == '__main__':
    def function(x):
        # A = np.array([
            # [1, 2, 3, 4],
            # [5, 6, 7, 8],
            # [9, 10, 11, 12],
            # [13, 14, 15, 16]
        # ])
        A = np.identity(4)
        return x.T.dot(A).dot(x) + 10

    x0 = np.array([1000, 98, 40, 10])
    fom = FOM(function, x0, 0.1, 0.001)
    print(list(map(lambda x: round(x, 1), fom.optimize())))
