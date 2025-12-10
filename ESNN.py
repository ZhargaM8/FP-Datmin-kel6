import torch
import torch.nn as nn
import snntorch as snn
from snntorch import spikegen
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
import copy

class EvolutionarySNN(BaseEstimator, ClassifierMixin):
    """
    Spiking Neural Network built with snnTorch, trained via Evolutionary Strategy.
    References Proposal: [cite: 202-205]
    """
    def __init__(self, n_inputs, n_outputs, n_hidden=64, pop_size=10, generations=10, beta=0.9, num_steps=25, device='cpu'):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_hidden = n_hidden
        self.pop_size = pop_size
        self.generations = generations
        self.beta = beta      # Decay rate for LIF neurons
        self.num_steps = num_steps # Temporal window size
        self.device = device
        self.best_model_state = None
        self.best_fitness = 0.0

    def _build_net(self):
        """Constructs a standard SNN architecture using snnTorch."""
        net = nn.Sequential(
            nn.Linear(self.n_inputs, self.n_hidden),
            snn.Leaky(beta=self.beta, init_hidden=True),
            nn.Linear(self.n_hidden, self.n_outputs),
            snn.Leaky(beta=self.beta, init_hidden=True, output=True)
        ).to(self.device)
        return net

    def _get_weights_from_net(self, net):
        """Extracts all weights/biases into a single 1D vector (genome)."""
        return torch.cat([p.data.view(-1) for p in net.parameters()])

    def _set_weights_to_net(self, net, weights_vector):
        """Loads a 1D weight vector back into the PyTorch network."""
        idx = 0
        for p in net.parameters():
            num_params = p.data.numel()
            # Slice the vector and reshape it to match the layer
            p.data.copy_(weights_vector[idx:idx+num_params].view(p.data.shape))
            idx += num_params

    def _evaluate_fitness(self, net, X, y):
        """Runs the SNN and calculates accuracy."""
        # Convert data to spike trains (Rate Coding)
        # Input X shape: (samples, features) -> Spikes: (time, samples, features)
        spike_data = spikegen.rate(X, num_steps=self.num_steps).to(self.device)
        targets = y.to(self.device)
        
        net.eval()
        with torch.no_grad():
            spk_rec = []
            utils_spk, _ = net(spike_data[0]) # Initialize
            
            # Forward pass over time steps
            for step in range(self.num_steps):
                spk_out, _ = net(spike_data[step])
                spk_rec.append(spk_out)
            
            # Sum spikes over time to get prediction (Rate decoding)
            spk_rec = torch.stack(spk_rec, dim=0)
            total_spikes = spk_rec.sum(dim=0) 
            _, preds = total_spikes.max(1)
            
            acc = (preds == targets).float().mean().item()
        return acc

    def fit(self, X, y):
        # Ensure y is numpy array if it's a pandas Series
        if hasattr(y, 'values'):
            y = y.values
            
        # Convert inputs to Tensor
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        
        # 1. Initialize Population
        template_net = self._build_net()
        param_count = sum(p.numel() for p in template_net.parameters())
        
        # Population: List of weight vectors
        population = [torch.randn(param_count).to(self.device) * 0.1 for _ in range(self.pop_size)]
        
        print(f"Initialized ESNN Population: {self.pop_size} agents, {param_count} parameters each.")

        # 2. Evolution Loop
        for gen in range(self.generations):
            fitness_scores = []
            
            for genome in population:
                self._set_weights_to_net(template_net, genome)
                acc = self._evaluate_fitness(template_net, X_tensor, y_tensor)
                fitness_scores.append(acc)
            
            # Track best
            max_fit = max(fitness_scores)
            best_idx = np.argmax(fitness_scores)
            
            if max_fit > self.best_fitness:
                self.best_fitness = max_fit
                self.best_model_state = population[best_idx].clone()
            
            print(f"ESNN Gen {gen+1}/{self.generations} - Best Acc: {self.best_fitness:.4f}")
            
            # Selection (Simple Elitism + Mutation)
            sorted_idx = np.argsort(fitness_scores)[::-1]
            survivors = [population[i] for i in sorted_idx[:self.pop_size//2]]
            
            new_pop = survivors[:]
            while len(new_pop) < self.pop_size:
                parent = survivors[np.random.randint(len(survivors))]
                child = parent + torch.randn_like(parent) * 0.05
                new_pop.append(child)
            
            population = new_pop

        return self

    def predict_proba(self, X):
        """Returns probability estimates (normalized spike counts)."""
        X_tensor = torch.FloatTensor(X).to(self.device)
        net = self._build_net()
        self._set_weights_to_net(net, self.best_model_state)
        
        spike_data = spikegen.rate(X_tensor, num_steps=self.num_steps).to(self.device)
        
        net.eval()
        with torch.no_grad():
            spk_rec = []
            for step in range(self.num_steps):
                spk_out, _ = net(spike_data[step])
                spk_rec.append(spk_out)
            
            spk_rec = torch.stack(spk_rec, dim=0)
            total_spikes = spk_rec.sum(dim=0)
            # Normalize to sum to 1 (Softmax-like)
            probs = torch.nn.functional.softmax(total_spikes.float(), dim=1)
            
        return probs.cpu().numpy()