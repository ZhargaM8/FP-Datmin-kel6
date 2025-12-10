import random
import numpy as np
from deap import base, creator, tools
from sklearn.model_selection import cross_val_score

class ASHOFeatureSelector:
    def __init__(self, classifier, n_agents=10, max_iter=20, random_state=42):
        self.classifier = classifier
        self.n_agents = n_agents
        self.max_iter = max_iter
        self.rng = np.random.default_rng(random_state)
        self.best_mask = None
        self.best_fitness = 0.0
        
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

    def _evaluate(self, individual, X, y):
        mask = individual > 0.5
        
        if not np.any(mask):
            mask[self.rng.integers(0, len(individual))] = True
            
        X_subset = X[:, mask]

        scores = cross_val_score(self.classifier, X_subset, y, cv=3, scoring='accuracy')
        return (scores.mean(),)

    def fit(self, X, y):
        n_features = X.shape[1]
        
        toolbox = base.Toolbox()
        toolbox.register("attr_float", self.rng.random)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n_features)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", self._evaluate, X=X, y=y)

        pop = toolbox.population(n=self.n_agents)
        
        fitnesses = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        best_ind = tools.selBest(pop, 1)[0]
        self.best_mask = best_ind.copy()
        self.best_fitness = best_ind.fitness.values[0]

        for t in range(self.max_iter):
            pop.sort(key=lambda ind: ind.fitness.values[0], reverse=True)
            elite = pop[0]
            
            next_pop = [creator.Individual(ind.copy()) for ind in pop]
            
            for i in range(self.n_agents):
                r1 = self.rng.random()
                r2 = self.rng.random()
                
                if r1 > 0.5:
                    x = self.rng.random() * 0.5
                    y_spiral = self.rng.random() * 0.5
                    z = self.rng.random() * 0.5
                    next_pop[i][:] = elite + (elite - pop[i]) * x * y_spiral * z
                else:
                    step = self.rng.normal(0, 1, n_features)
                    next_pop[i][:] = pop[i] + step * 0.01

                if r2 > 0.5:
                     next_pop[i][:] = next_pop[i] * 0.95 + elite * 0.05
                
                next_pop[i][:] = np.clip(next_pop[i], 0, 1)

            fitnesses = list(map(toolbox.evaluate, next_pop))
            for i, fit in enumerate(fitnesses):
                next_pop[i].fitness.values = fit
                
                if next_pop[i].fitness.values[0] > pop[i].fitness.values[0]:
                    pop[i][:] = next_pop[i][:]
                    pop[i].fitness.values = next_pop[i].fitness.values

            current_best = tools.selBest(pop, 1)[0]
            if current_best.fitness.values[0] > self.best_fitness:
                self.best_fitness = current_best.fitness.values[0]
                self.best_mask = current_best.copy()
            
            print(f"ASHO Iteration {t+1}/{self.max_iter} - Best Fitness: {self.best_fitness:.4f}")

        return self

    def transform(self, X):
        mask = self.best_mask > 0.5
        if not np.any(mask):
            mask[0] = True
        return X[:, mask]