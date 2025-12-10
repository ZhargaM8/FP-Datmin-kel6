import random
import numpy as np
from deap import base, creator, tools
from sklearn.model_selection import cross_val_score

class ASHOFeatureSelector:
    """
    Adaptive Sea Horse Optimization using DEAP framework.
    References Proposal: [cite: 107-108, 198]
    """
    def __init__(self, classifier, n_agents=10, max_iter=20, random_state=42):
        self.classifier = classifier
        self.n_agents = n_agents
        self.max_iter = max_iter
        self.rng = np.random.default_rng(random_state)
        self.best_mask = None
        self.best_fitness = 0.0
        
        # Setup DEAP
        # 1. Define Optimization Problem (Maximize Accuracy)
        # We check if it exists to avoid errors on re-imports in Jupyter
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

    def _evaluate(self, individual, X, y):
        """Fitness function: Accuracy of the classifier on selected features."""
        # Convert continuous individual (0.0-1.0) to binary mask
        mask = individual > 0.5
        
        # If no features selected, select at least one random feature to avoid error
        if not np.any(mask):
            mask[self.rng.integers(0, len(individual))] = True
            
        X_subset = X[:, mask]
        
        # 3-Fold Cross Validation
        scores = cross_val_score(self.classifier, X_subset, y, cv=3, scoring='accuracy')
        return (scores.mean(),)

    def fit(self, X, y):
        n_features = X.shape[1]
        
        toolbox = base.Toolbox()
        # Attribute generator: float between 0 and 1
        toolbox.register("attr_float", self.rng.random)
        # Structure initializers
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n_features)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", self._evaluate, X=X, y=y)

        # 1. Initialize Population
        pop = toolbox.population(n=self.n_agents)
        
        # Evaluate initial fitness
        fitnesses = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
            
        # Track global best
        best_ind = tools.selBest(pop, 1)[0]
        self.best_mask = best_ind.copy()
        self.best_fitness = best_ind.fitness.values[0]

        # 2. ASHO Main Loop
        for t in range(self.max_iter):
            # Sort population (Sea horses have social hierarchy)
            pop.sort(key=lambda ind: ind.fitness.values[0], reverse=True)
            elite = pop[0] # The best sea horse
            
            # Create next generation container
            next_pop = [creator.Individual(ind.copy()) for ind in pop]
            
            for i in range(self.n_agents):
                r1 = self.rng.random()
                r2 = self.rng.random()
                
                # --- Movement Logic (Custom Operator) ---
                # We operate on numpy arrays directly since DEAP Individuals are numpy arrays
                
                if r1 > 0.5:
                    # Spiral Motion (Exploitation)
                    x = self.rng.random() * 0.5
                    y_spiral = self.rng.random() * 0.5
                    z = self.rng.random() * 0.5
                    # Update position
                    next_pop[i][:] = elite + (elite - pop[i]) * x * y_spiral * z
                else:
                    # Brownian Motion (Exploration)
                    step = self.rng.normal(0, 1, n_features)
                    next_pop[i][:] = pop[i] + step * 0.01

                # --- Predation Logic ---
                if r2 > 0.5:
                     next_pop[i][:] = next_pop[i] * 0.95 + elite * 0.05
                
                # Clip values to stay valid (0-1)
                next_pop[i][:] = np.clip(next_pop[i], 0, 1)

            # Evaluate new population
            fitnesses = list(map(toolbox.evaluate, next_pop))
            for i, fit in enumerate(fitnesses):
                next_pop[i].fitness.values = fit
                
                # Greedy Selection: Only keep if better
                if next_pop[i].fitness.values[0] > pop[i].fitness.values[0]:
                    pop[i][:] = next_pop[i][:]
                    pop[i].fitness.values = next_pop[i].fitness.values

            # Update Global Best
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