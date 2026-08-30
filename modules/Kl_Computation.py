import numpy as np
import torch
import torch.nn.functional as F

def compute_exact_kl_divergence(model_k, model_k_minus_1, S_batch, calc_stats_fn, k, num_classes=10):
    """
    [Exact Analytical Version]
    Leveraging the fact that T is a finite discrete variable (e.g., 10 classes),
    this function computes the exact conditional KL divergence D_KL(P_k || P_{k-1}),
    completely bypassing the need for MCMC or Annealed Importance Sampling (AIS).
    """
    batch_size = S_batch.shape[0]
    device = S_batch.device
    
    energies_k = []
    energies_k_minus_1 = []
    
    with torch.no_grad():
        for c in range(num_classes):
            T_dummy = torch.zeros(batch_size, num_classes, device=device)
            T_dummy[:, c] = 1.0
            
            # Consolidate function calls into a single execution for computational efficiency
            stats_dict = calc_stats_fn(T_dummy, S_batch)
            
            # Sufficient statistics and energy for the k-th order model
            stats_k = stats_dict[k]
            e_k = torch.matmul(stats_k, model_k.theta)
            energies_k.append(e_k)
            
            # Sufficient statistics and energy for the (k-1)-th order model
            if k - 1 == 0:
                # Use T_dummy directly as the 0-th order sufficient statistic
                stats_k_minus_1 = T_dummy 
            else:
                stats_k_minus_1 = stats_dict[k - 1]
                
            e_k_minus_1 = torch.matmul(stats_k_minus_1, model_k_minus_1.theta)
            energies_k_minus_1.append(e_k_minus_1)
            
        E_k_tensor = torch.stack(energies_k, dim=1)
        E_k_minus_1_tensor = torch.stack(energies_k_minus_1, dim=1)
        
        # Calculate log partition functions (log Z)
        log_Z_k = torch.logsumexp(E_k_tensor, dim=1)
        log_Z_k_minus_1 = torch.logsumexp(E_k_minus_1_tensor, dim=1)
        
        # Calculate conditional probability p(T|S) for the k-th order model
        p_k_given_S = F.softmax(E_k_tensor, dim=1)
        
        # Expected energy difference: E_{p_k}[E_k - E_{k-1}]
        energy_diff = E_k_tensor - E_k_minus_1_tensor
        expected_energy_diff = torch.sum(p_k_given_S * energy_diff, dim=1)
        
        # Exact KL divergence computation
        kl_S = expected_energy_diff - log_Z_k + log_Z_k_minus_1
        
        return kl_S.mean().item()

def compute_ais_log_weight_mala(x_init, energy_base, energy_target, num_steps, step_size, device):
    """
    Approximates the log importance weights by annealing from a base model $p^{(k-1)}$ 
    to a target model $p^{(k)}$ using the Metropolis-Adjusted Langevin Algorithm (MALA) 
    for MCMC transitions. This is essential for estimating the intractable partition function ratio.
    
    Args:
        x_init (torch.Tensor): Initial samples from the base model, shape [batch_size, dim].
        energy_base (callable): Function returning the energy E_base(x) of the base model.
        energy_target (callable): Function returning the energy E_target(x) of the target model.
        num_steps (int): Total number of annealing steps (N).
        step_size (float): Step size for the Langevin dynamics.
        device (torch.device): Computation device.
        
    Returns:
        log_w (torch.Tensor): Accumulated log importance weights for each sample.
        x (torch.Tensor): Final samples drawn from the target distribution.
    """
    batch_size = x_init.shape[0]
    log_w = torch.zeros(batch_size, device=device)
    x = x_init.clone()
    
    # Define the annealing schedule for inverse temperature beta (from 0.0 to 1.0)
    betas = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
    
    for i in range(1, num_steps + 1):
        beta_prev = betas[i-1]
        beta_curr = betas[i]
        
        # ---------------------------------------------------------
        # 1. Log Weight Update
        # Accumulate log(f_curr / f_prev) = E_prev - E_curr
        # ---------------------------------------------------------
        # Evaluate the energy functions for the current state x yielding a vector of shape (batch_size,)
        e_base_x = energy_base(x)
        e_target_x = energy_target(x)
        
        e_prev = (1 - beta_prev) * e_base_x + beta_prev * e_target_x
        e_curr = (1 - beta_curr) * e_base_x + beta_curr * e_target_x
        
        log_w += (e_prev - e_curr)
        
        # ---------------------------------------------------------
        # 2. MCMC Transition via MALA (Metropolis-Adjusted Langevin Algorithm)
        # ---------------------------------------------------------
        if i < num_steps:
            # Compute the gradient of the intermediate energy function at the current state
            x.requires_grad_(True)
            e_x = (1 - beta_curr) * energy_base(x) + beta_curr * energy_target(x)
            grad_e = torch.autograd.grad(e_x.sum(), x)[0]
            
            with torch.no_grad():
                # Sample a new proposed state x_prop according to the Langevin proposal distribution q(x'|x)
                noise = torch.randn_like(x)
                x_prop = x - 0.5 * step_size * grad_e + torch.sqrt(torch.tensor(step_size, device=device)) * noise
                
            # Compute the gradient of the intermediate energy function at the proposed state x_prop
            x_prop.requires_grad_(True)
            e_x_prop = (1 - beta_curr) * energy_base(x_prop) + beta_curr * energy_target(x_prop)
            grad_e_prop = torch.autograd.grad(e_x_prop.sum(), x_prop)[0]
            
            with torch.no_grad():
                # Compute log q(x_prop | x). The normalizing constant is omitted as it cancels out in the acceptance ratio.
                log_q_prop_given_x = -0.5 * torch.sum((x_prop - x + 0.5 * step_size * grad_e)**2, dim=1) / step_size
                
                # Compute log q(x | x_prop), representing the reverse transition probability.
                log_q_x_given_prop = -0.5 * torch.sum((x - x_prop + 0.5 * step_size * grad_e_prop)**2, dim=1) / step_size
                
                # Compute the log acceptance probability (log_alpha).
                # log_alpha = -E(x_prop) + E(x) + log q(x | x_prop) - log q(x_prop | x)
                log_alpha = -e_x_prop + e_x + log_q_x_given_prop - log_q_prop_given_x
                
                # Execute the Metropolis-Hastings accept/reject step
                u = torch.rand(batch_size, device=device)
                accept_mask = torch.log(u) < log_alpha
                
                # Update state x to x_prop only for the accepted samples
                x = torch.where(accept_mask.unsqueeze(1), x_prop, x)
                
    return log_w, x

def compute_kl_divergence(log_w, x_final, energy_base, energy_target):
    """
    Computes the exact KL divergence $D_{KL}(p^{(k)} || p^{(k-1)})$ using the final samples 
    and log weights obtained from Annealed Importance Sampling.
    
    Args:
        log_w (torch.Tensor): Log importance weights from AIS.
        x_final (torch.Tensor): Final MCMC samples representing the target distribution.
        energy_base (callable): Function returning the energy of the base model.
        energy_target (callable): Function returning the energy of the target model.
        
    Returns:
        kl_div (float): The computed KL divergence.
        log_z_ratio (float): The estimated log partition function ratio.
    """
    N = log_w.shape[0]
    
    # 1. Log partition function ratio: log(Z_target / Z_base) = log(mean(exp(log_w)))
    # Utilize logsumexp for numerical stability to prevent overflow/underflow.
    log_z_ratio = torch.logsumexp(log_w, dim=0) - np.log(N)
    
    # 2. Expected energy difference term. x_final serves as valid samples from the target distribution $p^{(k)}$.
    # Expected value corresponds to E_p^{(k)}[E_base(x) - E_target(x)].
    e_diff = energy_base(x_final) - energy_target(x_final)
    expected_diff = torch.mean(e_diff)
    
    # 3. Final KL divergence computation.
    # Note: Since log_z_ratio computes log(Z_target / Z_base), it must be subtracted.
    kl_div = expected_diff - log_z_ratio
    
    return kl_div.item(), log_z_ratio.item()