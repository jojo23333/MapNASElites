from typing import DefaultDict
from numpy.core.records import array
from time import time
import copy
import numpy as np
from utils import random_spec, mutate_spec, get_api

MAX_N = 6
MAX_R = 8 
MAX_L = 6
MAX_S = 6
NASBENCH_DIM = (MAX_N, MAX_R, MAX_L, MAX_S)

def dfs(adj, cur_node, cur_len, max_len, min_len):
    num_routes = 0
    # start = 0
    # end = adj.shape[0] - 1
    if cur_node == adj.shape[0] - 1:
        max_len = max(max_len, cur_len)
        min_len = min(min_len, cur_len)
        return 1, max_len, min_len
    
    for to, flag in enumerate(adj[cur_node]):
        if flag:
            res = dfs(adj, to, cur_len+1, max_len, min_len)
            num_routes += res[0]
            max_len = max(max_len, res[1])
            min_len = min(min_len, res[2])
    return num_routes, max_len, min_len


# num_node 2~7
# num_routes 1~8
# max_len 1~6
# min_len 1~6
def get_nasbench_elites_3d(elite):
    adj = elite["module_adjacency"]
    modules = elite["module_operations"]
    num_routes, max_len, min_len = dfs(adj, 0, 0, 0, 10)
    return num_routes - 1, max_len - 1, min_len - 1

def get_nasbench_elites(elite):
    adj = elite["module_adjacency"]
    modules = elite["module_operations"]
    num_routes, max_len, min_len = dfs(adj, 0, 0, 0, 10)
    num_node = len(modules) 
    return num_node - 2, num_routes - 1, max_len - 1, min_len - 1

NASBENCH_PARAM_DIST = [227274, 2538506, 3989898, 6230410, 13126538, 50000000]
def get_nasbench_elites_params(elite):
    adj = elite["module_adjacency"]
    modules = elite["module_operations"]
    num_param = elite["trainable_parameters"]
    num_routes, max_len, min_len = dfs(adj, 0, 0, 0, 10)
    for i in range(len(NASBENCH_PARAM_DIST)):
        if num_param < NASBENCH_PARAM_DIST[i]:
            param_num_level = i
            return param_num_level - 1, num_routes - 1, max_len - 1, min_len - 1
    
# class EliteMatrix:
#     def __init__(self, dim1, dim2, dim3, elites=None):
#         self.dims = [dim1, dim2, dim3]
#         if elites is None:
#             self.elites = [
#                     [
#                         [None] * dim3 
#                         for i in range(dim2)
#                     ]
#                     for i in range(dim1)
#                 ]
#         else:
#             self.elites = elites
    
#     def update_elites(self, elite):
#         adj = elite["module_adjacency"]
#         num_routes, max_len, min_len = dfs(adj, 0, 0, 0, 10)
#         assert num_routes > 0 and max_len > 0 and min_len > 0
#         x, y, z = num_routes - 1, max_len - 1, min_len - 1
#         cur_elite = self.elites[x][y][z]
#         if cur_elite is None:
#             self.elites[x][y][z] = copy.deepcopy(elite)
#         elif cur_elite['validation_accuracy'] < elite['validation_accuracy']:
#             self.elites[x][y][z] = copy.deepcopy(elite)
            
#     def get_performance(self, key='validation_accuracy'):
#         return np.array([
#             [
#                 [
#                     x[key] if x is not None and key in x else 0 
#                     for x in y
#                 ]
#                 for y in z
#             ]
#             for z in self.elites
#         ])
    
#     def iter(self):
#         for i in range(self.dims[0]):
#             for j in range(self.dims[1]):
#                 for k in range(self.dims[2]):
#                     if self.elites[i][j][k] is not None:
#                         yield self.elites[i][j][k], (i, j, k)
    
#     def update_metric(self, metric, x, y, z, key='new_metric'):
#         assert self.elites[x][y][z] is not None
#         self.elites[x][y][z][key] = metric
    
#     @classmethod
#     def load(cls, x):
#         elites = np.load(x, allow_pickle=True)
#         dims = elites.shape
#         return EliteMatrix(*dims, elites.tolist())

#     def save(self, path):
#         np.save(path, self.elites)

class EliteMatrix:
    """
    A class for saving qulity diversity policies, capable of handling different dim size
    """
    def __init__(self, dims, elites=None, get_idx_fn=get_nasbench_elites_3d):
        self.dims = dims
        self.get_idx_fn = get_idx_fn
        if elites is None:
            self.elites = np.full(dims, None)
        else:
            self.elites = elites
    
    def update_elites(self, elite):
        
        idx = self.get_idx_fn(elite)
        
        cur_elite = self.elites[idx]
        if cur_elite is None:
            self.elites[idx] = copy.deepcopy(elite)
        elif cur_elite['validation_accuracy'] < elite['validation_accuracy']:
            self.elites[idx] = copy.deepcopy(elite)
            
    def get_performance(self, key='validation_accuracy'):
        fn = lambda x: x[key] if x is not None and key in x else 0 
        return np.array([fn(x) for x in self.elites.flatten()]).reshape(self.dims)
    
    def iter(self):
        for idx, x in np.ndenumerate(self.elites):
            if self.elites[idx] is not None:
                yield idx, self.elites[idx]
    
    def sorted_iter(self):
        list_with_idx = list(np.ndenumerate(self.elites))
        list_with_idx = [x for x in list_with_idx if x[1] is not None]
        sorted_list = sorted(list_with_idx, key=lambda x: x[1]['validation_accuracy'] if x is not None else 0)
        # print([self.elites[x[0]]['validation_accuracy'] for x in sorted_list[::-1]])
        # print(self.get_performance())
        return sorted_list[::-1]
    
    def update_metric(self, metric, idx, key='new_metric'):
        assert self.elites[idx] is not None
        self.elites[idx][key] = metric
    
    @classmethod
    def load(cls, x):
        elites = np.load(x, allow_pickle=True)
        dims = elites.shape
        return cls(dims, elites)

    def save(self, path):
        np.save(path, self.elites)



class SimpleGAController:
    def __init__(self, num_population=1001, num_individual=20, sigma=0.02) -> None:
        self.num_pop = num_population
        self.num_top_ind = num_individual
        self.sigma = sigma
    
        self.population = None
        self.elite = None
        self.fitness = []
        self.nasbench = get_api()

    def create_population(self):
        """
        Creates the initial population of the genetic algorithm as a list of networks' weights (i.e. solutions). Each element in the list holds a different weights of the PyTorch model.
        The method returns a list holding the weights of all solutions.
        """

        initial_spec = random_spec()
        spec_population = [initial_spec]
        for idx in range(self.num_pop-1):
            spec_population.append(mutate_spec(initial_spec))
        return spec_population

    def eval_population(self, population, num_runs=1):
        pop_with_fitness = []
        for p in population:
            # TODO here
            f = {
                "validation_accuracy": 0,
                "test_accuracy": 0,
                'training_time': 0
            }
            for i in range(num_runs):
                f_i =  self.get_fitness(spec=p)
                f["validation_accuracy"] += f_i["validation_accuracy"]
                f["test_accuracy"] += f_i["test_accuracy"]
                f["training_time"] += f_i["training_time"]
            f = {k: v/num_runs for k, v in f.items()}
            pop_with_fitness.append((p, f))
        pop_with_fitness = sorted(pop_with_fitness, key=lambda x:-x[1]['validation_accuracy'])
        return pop_with_fitness

    def get_fitness(self, spec):
        data = self.nasbench.query(spec)
        time_spent, _ = self.nasbench.get_budget_counters()
        # data['time_spent'] = time_spent
        return data

    def evolve(self, num_generations, log=False):
        best_val_acc = 0
        best_test_acc = 0
        for g in range(num_generations):
            starttime = time()
            if g == 0:
                next_generations = self.create_population()
            else:
                next_generations = []
                for i in range(self.num_pop-1):
                    k = np.random.choice(range(self.num_top_ind))
                    next_generations.append(mutate_spec(self.population[k]))
            evaled_population = self.eval_population(next_generations)
            
            if g == 0:
                self.elite_candidates = [x[0] for x in evaled_population[:10]]
            else:
                self.elite_candidates = [x[0] for x in evaled_population[:9]] + [self.elite[0]]

            elites = self.eval_population(self.elite_candidates, num_runs=10)

            if self.elite is not None:
                del self.elite
            self.elite = elites[0]
            if self.population is not None:
                del self.population
            self.population = [x[0] for x in evaled_population]

            hist = np.histogram(
                a = [x[1]["validation_accuracy"] for x in elites],
                bins = 10,
                range = (0.8, 1),
            )
            if log:
                wandb.log({
                    'elite_val': elites[0][1]['validation_accuracy'],
                    'elite_test': elites[0][1]['test_accuracy'],
                    'pop_histogram': wandb.Histogram(np_histogram=hist)
                })
            if elites[0][1]['validation_accuracy'] > best_val_acc:
                best_val_acc = elites[0][1]['validation_accuracy']
                best_test_acc = elites[0][1]['test_accuracy']
            endtime = time()
            if log:
                print(
                    f"Generation {g} Finished, {(endtime - starttime):.2f}s\n Elite score: {elites[0][1]['test_accuracy']}")

        return self.elite, best_val_acc, best_test_acc

    