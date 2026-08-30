import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import combinations

class InteractionSufficientStatistics:
    def __init__(self, max_order=5):
        """
        max_order: Maximum order of interactions (limited to 5 by default)
        """
        self.max_order = max_order
        self._cached_num_sources = None
        self._combinations_by_order = {}

    def _update_combinations(self, num_sources):
        if self._cached_num_sources != num_sources:
            limit_order = min(self.max_order, num_sources)
            self._combinations_by_order = {
                k: list(combinations(range(num_sources), k)) 
                for k in range(1, limit_order + 1)
            }
            self._cached_num_sources = num_sources

    def __call__(self, T, S):
        """
        T: One-hot tensor of prediction results (batch_size, num_classes)
        S: Tensor of source variables (batch_size, num_sources)
        """
        num_sources = S.shape[1]
        self._update_combinations(num_sources)
        
        # Store the 0th-order term (T alone) as the initial state
        cumulative_stats = [T] 
        stats_for_models = {0: T}
        
        for k in self._combinations_by_order.keys():
            order_k_stats = []
            for idx in self._combinations_by_order[k]:
                interaction_term = S[:, idx].prod(dim=1, keepdim=True)
                order_k_stats.append(T * interaction_term)
            
            cumulative_stats.append(torch.cat(order_k_stats, dim=1))
            stats_for_models[k] = torch.cat(cumulative_stats, dim=1)
            
        return stats_for_models

class ExponentialFamilyModel(nn.Module):
    def __init__(self, stats_dim):
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(stats_dim))

    def sample_T_given_S(self, S_batch, calc_stats_fn, k):
        """
        Sample T from the conditional distribution p(T|S; θ) based on the current parameters θ.
        """
        batch_size = S_batch.shape[0]
        num_classes = 10
        device = S_batch.device
        
        energies = []

        with torch.no_grad():
            for c in range(num_classes):
                T_dummy = torch.zeros(batch_size, num_classes, device=device)
                T_dummy[:, c] = 1.0
                
                stats_c = calc_stats_fn(T_dummy, S_batch)[k]
                energy_c = torch.matmul(stats_c, self.theta)
                energies.append(energy_c)
                
            energies_tensor = torch.stack(energies, dim=1) # (batch, 10)
            
            # compute p(T|S)
            p_T_given_S = F.softmax(energies_tensor, dim=1)
            
            # Sample T from the categorical distribution (MCMC step)
            sampled_class_indices = torch.multinomial(p_T_given_S, num_samples=1).squeeze()
            
            # Convert the sampled T into a one-hot vector
            T_sampled = F.one_hot(sampled_class_indices, num_classes=10).float()
            
        return T_sampled

    def compute_negative_stats_mean(self, S_batch, calc_stats_fn, k):
        """
        Compute sufficient statistics for all 10 classes and precisely calculate the expectation weighted by Softmax probabilities.
        """
        batch_size = S_batch.shape[0]
        num_classes = 10
        device = S_batch.device
        
        stats_all_classes = []
        energies = []
        
        # 1. Calculate statistics and energy E_k(T=c, S) for all 10 classes
        for c in range(num_classes):
            T_dummy = torch.zeros(batch_size, num_classes, device=device)
            T_dummy[:, c] = 1.0
            
            stats_c = calc_stats_fn(T_dummy, S_batch)[k]
            stats_all_classes.append(stats_c)
            
            energy_c = torch.matmul(stats_c, self.theta)
            energies.append(energy_c)
            
        energies_tensor = torch.stack(energies, dim=1) # (batch, 10)
        
        # 2. Calculate conditional probability p(T|S) using Softmax
        p_T_given_S = F.softmax(energies_tensor, dim=1)
        
        # 3. Calculate expectation E_{p(T|S)}[\phi(T, S)] weighted by probability
        expected_stats = torch.zeros_like(stats_all_classes[0])
        for c in range(num_classes):
            prob_c = p_T_given_S[:, c].unsqueeze(1)
            expected_stats += prob_c * stats_all_classes[c]
            
        # Take the average over the mini-batch direction E_{p_data(S)}[...]
        return expected_stats.mean(dim=0)

    def likelihood_step(self, optimizer, pos_mean, neg_mean):
        """
        Update parameters based on the difference between the sample mean of real data and the mean of MCMC samples
        """
        # Gradient = <φ>_data - <φ>_model
        grad_constant = (pos_mean - neg_mean).detach()
        surrogate_loss = -torch.dot(self.theta, grad_constant)
        
        optimizer.zero_grad()
        surrogate_loss.backward()
        optimizer.step()
        
        return surrogate_loss.item()
