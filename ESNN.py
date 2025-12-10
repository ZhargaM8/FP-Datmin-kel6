import torch
import torch.nn as nn
import snntorch as snn
from snntorch import spikegen
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
import copy

class EvolutionarySNN(BaseEstimator, ClassifierMixin):
    def __init__(self, n_inputs, n_outputs, n_hidden=64, pop_size=10, generations=10, beta=0.9, num_steps=25, device='cpu'):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_hidden = n_hidden
        self.pop_size = pop_size
        self.generations = generations
        self.beta = beta
        self.num_steps = num_steps
        self.device = device
        self.best_model_state = None
        self.best_fitness = 0.0

    def _build_net(self):
        net = nn.Sequential(
            nn.Linear(self.n_inputs, self.n_hidden),
            snn.Leaky(beta=self.beta, init_hidden=True),
            nn.Linear(self.n_hidden, self.n_outputs),
            snn.Leaky(beta=self.beta, init_hidden=True, output=True)
        ).to(self.device)
        return net

    def _get_weights_from_net(self, net):
        return torch.cat([p.data.view(-1) for p in net.parameters()])

    def _set_weights_to_net(self, net, weights_vector):
        idx = 0
        for p in net.parameters():
            num_params = p.data.numel()
            p.data.copy_(weights_vector[idx:idx+num_params].view(p.data.shape))
            idx += num_params

    def _evaluate_fitness(self, net, X, y):
        spike_data = spikegen.rate(X, num_steps=self.num_steps).to(self.device)
        targets = y.to(self.device)
        
        net.eval()
        with torch.no_grad():
            spk_rec = []
            utils_spk, _ = net(spike_data[0])
            
            for step in range(self.num_steps):
                spk_out, _ = net(spike_data[step])
                spk_rec.append(spk_out)
            
            spk_rec = torch.stack(spk_rec, dim=0)
            total_spikes = spk_rec.sum(dim=0) 
            _, preds = total_spikes.max(1)
            
            acc = (preds == targets).float().mean().item()
        return acc

    def fit(self, X, y):
        if hasattr(y, 'values'):
            y = y.values

        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        
        template_net = self._build_net()
        param_count = sum(p.numel() for p in template_net.parameters())
        
        population = [torch.randn(param_count).to(self.device) * 0.1 for _ in range(self.pop_size)]
        
        print(f"Initialized ESNN Population: {self.pop_size} agents, {param_count} parameters each.")

        for gen in range(self.generations):
            fitness_scores = []
            
            for genome in population:
                self._set_weights_to_net(template_net, genome)
                acc = self._evaluate_fitness(template_net, X_tensor, y_tensor)
                fitness_scores.append(acc)
            
            max_fit = max(fitness_scores)
            best_idx = np.argmax(fitness_scores)
            
            if max_fit > self.best_fitness:
                self.best_fitness = max_fit
                self.best_model_state = population[best_idx].clone()
            
            print(f"ESNN Gen {gen+1}/{self.generations} - Best Acc: {self.best_fitness:.4f}")
            
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
            probs = torch.nn.functional.softmax(total_spikes.float(), dim=1)
            
        return probs.cpu().numpy()