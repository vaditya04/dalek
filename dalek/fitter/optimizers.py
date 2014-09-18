from abc import ABCMeta, abstractmethod

from dalek.parallel import ParameterCollection
import numpy as np
import random

class BaseOptimizer(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class RandomSampling(BaseOptimizer):
    def __init__(self, fitter_configuration):
        self.fitter_configuration = fitter_configuration
        self.lbounds = self.fitter_configuration.lbounds
        self.ubounds = self.fitter_configuration.ubounds
        self.n = self.fitter_configuration.number_of_samples

    def __call__(self, parameter_collection):
        parameters = []
        for _ in range(self.n):
            parameters.append(np.random.uniform(lbounds, ubounds))
        return ParameterCollection(np.array(parameters), columns=self.fitter_configuration.parameter_names)

class NoiseMeasurement(BaseOptimizer):
    def __init__(self, fitter_configuration):
        self.fitter_configuration = fitter_configuration
        self.n = self.fitter_configuration.number_of_samples

    def __call__(self, parameter_collection):
        parameters = [random.randint(0, 2**16) for _ in range(self.n)]
        return ParameterCollection(np.array(parameters), columns=['montecarlo.seed'])

class LuusJaakolaOptimizer(BaseOptimizer):
    def __init__(self, fitter_configuration):
        self.fitter_configuration = fitter_configuration
        self.x = (self.fitter_configuration.lbounds +
                  self.fitter_configuration.ubounds) * 0.5
        self.n = fitter_configuration.number_of_samples
        self.d = np.array(self.fitter_configuration.ubounds -
                          self.fitter_configuration.lbounds) * 0.5

    def __call__(self, parameter_collection):
        best_fit = parameter_collection.ix[parameter_collection['dalek.fitness'].argmin()]
        best_x = best_fit.values[:-1]
        new_parameters = [best_x]
        lbounds = self.fitter_configuration.lbounds
        ubounds = self.fitter_configuration.ubounds
        for _ in range(self.n):
            new_parameters.append(np.random.uniform(np.clip(best_x - self.d, lbounds, ubounds),
                                                    np.clip(best_x + self.d, lbounds, ubounds)))
        self.d *= 0.95
        return ParameterCollection(np.array(new_parameters), columns=self.fitter_configuration.parameter_names)

class DEOptimizer(BaseOptimizer):
    def __init__(self, fitter_configuration):
        self.fitter_configuration = fitter_configuration
        self.population = None
        self.cr = self.fitter_configuration.optimizer_configuration.get('cr', 0.9)
        self.f = self.fitter_configuration.optimizer_configuration.get('f', 0.5)
        self.n = fitter_configuration.number_of_samples
        self.lbounds = np.array(self.fitter_configuration.lbounds)
        self.ubounds = np.array(self.fitter_configuration.ubounds)

    def violates_bounds(self, x):
        return any(x < self.lbounds) or any(x > self.ubounds)
        
    def __call__(self, parameter_collection):
        if self.population is None:
            self.population = np.array(parameter_collection.values)
        else:
            new_population = np.array(parameter_collection.values)
            for index, vector in enumerate(self.population):
                if new_population[index][-1] < vector[-1]:
                    self.population[index] = np.array(new_population[index])
        d = len(self.population[0]) - 1
        candidates = np.array([vec[:-1] for vec in self.population])
        for index, vector in enumerate(self.population):
            indices = [i for i in range(self.n) if i != index]
            random.shuffle(indices)
            i1, i2, i3 = indices[:3]
            a, b, c = self.population[i1], self.population[i2], self.population[i3]
            r_ = random.randint(0, d - 1)
            for j, x in enumerate(vector[:-1]):
                ri = random.random()
                if ri < self.cr or j == r_:
                    candidates[index][j] = a[j] + self.f * (b[j] - c[j])
                else:
                    candidates[index][j] = x
            if self.violates_bounds(candidates[index]):
                candidates[index] = np.array(vector[:-1])
	params = ParameterCollection(np.array(candidates), columns=self.fitter_configuration.parameter_names)
	return params

class PSOOptimizerGbest(BaseOptimizer):
    def __init__(self, fitter_configuration):
        self.fitter_configuration = fitter_configuration
        self.x = None
        self.px = None
        self.v = None
        self.c1 = 2.05
        self.c2 = 2.05
        self.chi = 2.0 / (self.c1 + self.c2 - 2.0 + np.sqrt((self.c1 + self.c2) ** 2 - 4.0 * (self.c1 + self.c2)))
        self.n = fitter_configuration.number_of_samples
        self.lbounds = np.array(self.fitter_configuration.lbounds)
        self.ubounds = np.array(self.fitter_configuration.ubounds)        

    def neighbourhood(self, index):
        return [i for i in range(self.n) if i != index]

    def violates_bounds(self, x):
        return any(x < self.lbounds) or any(x > self.ubounds)
        
    def __call__(self, parameter_collection):
        candidates = np.array([x[:-1] for x in parameter_collection.values])
        if self.x is None:
            self.x = np.array(candidates)
            self.y = np.array([x[-1] for x in parameter_collection.values])
            self.px = np.array(candidates)
            self.py = np.array([x[-1] for x in parameter_collection.values])
            self.v = np.zeros(self.x.shape)
        else:
            for index, candidate in enumerate(parameter_collection.values):
                if candidate[-1] < self.py[index]:
                    self.px[index] = np.array(candidate[:-1])
                    self.py[index] = candidate[-1]
        gx = np.zeros(self.x.shape)
        for index, _ in enumerate(self.x):
            neighbours = self.neighbourhood(index)
            nx, ny = min(zip([self.px[i] for i in neighbours], 
                             [self.py[i] for i in neighbours]), 
                         key=lambda pair: pair[1])
            gx[index] = np.array(nx)
        self.v = self.chi * (self.v + 
                             self.c1 * np.random.sample(self.x.shape) * (self.px - self.x) + 
                             self.c2 * np.random.sample(self.x.shape) * (gx - self.x))
        candidates = np.array(self.x + self.v)
        for index, x in enumerate(candidates):
            if not self.violates_bounds(x):
                candidates[index] = np.array(x)
            else:
                candidates[index] = np.array(self.px[index])
        self.x += self.v
        params = ParameterCollection(candidates, columns=self.fitter_configuration.parameter_names)
        return params
        

optimizer_dict = {'random_sampling': RandomSampling,
                  'luus_jaakola': LuusJaakolaOptimizer,
                  'devolution': DEOptimizer,
                  'pso': PSOOptimizerGbest}
