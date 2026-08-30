import torch
import torch.nn.functional as F
import numpy as np
import random

# Import from existing custom modules
from modules.Parameter_Estimation import ExponentialFamilyModel
from modules.Kl_Computation import compute_exact_kl_divergence

def set_seed(seed):
    """Set random seeds for PyTorch, NumPy, and Python built-in random module."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train_one_epoch(model, train_loader, optimizer, criterion, device):
    """Trains the model for one epoch and returns the average loss."""
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)

def evaluate_and_extract_activations(model, test_loader, device, target_layers):
    """Evaluates accuracy on test data and extracts activations and predictions for each layer."""
    layer_data = {layer: [] for layer in target_layers}
    predictions = []
    correct, total = 0, 0
    
    model.eval() 
    with torch.no_grad(): 
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted_labels = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted_labels == labels).sum().item()
            
            activations = model.get_activations(images)
            for layer in target_layers:
                layer_data[layer].append(activations[layer])
            predictions.append(activations['predicted'])
            
    layer_activations = {layer: np.vstack(layer_data[layer]) for layer in target_layers}
    predictions = np.concatenate(predictions)
    test_acc = 100.0 * correct / total
    
    return test_acc, layer_activations, predictions

def scale_activations(S_all):
    """Scales activations by excluding L2 norm outliers (Note: intended for ReLU)."""
    norms = np.linalg.norm(S_all, axis=1)
    upper_bound = np.percentile(norms, 90)
    valid_norms = norms[(norms <= upper_bound)]
    scale_factor = valid_norms.mean() + 1e-8
    return S_all / scale_factor

def initialize_cdm_layer(layer_name, S_all, T_all, epoch_for_print, cdm_models, cdm_optimizers, 
                         is_initialized, calc_stats, max_order, device, lr=0.0001):
    """Calculates statistics and sets up the 0-th order model."""
    print(f"--- Starting CDM & KL Computation for {layer_name} (Epoch {epoch_for_print}) ---")
    print(f"[{layer_name}] Activation L2 Norm: {np.linalg.norm(S_all, axis=1).mean():.4f}")
    
    num_samples = len(S_all)
    cdm_batch_size = 512
    global_pos_mean = {k: 0.0 for k in range(max_order + 1)}
    
    with torch.no_grad():
        for i in range(0, num_samples, cdm_batch_size):
            S_batch = torch.tensor(S_all[i:i+cdm_batch_size], dtype=torch.float32).to(device)
            T_batch = torch.tensor(T_all[i:i+cdm_batch_size], dtype=torch.long).to(device)
            T_onehot = F.one_hot(T_batch, num_classes=10).float()
            
            pos_stats = calc_stats(T_onehot, S_batch)
            pos_stats[0] = T_onehot
            for k in range(max_order + 1):
                global_pos_mean[k] += pos_stats[k].sum(dim=0)
                
        for k in range(max_order + 1):
            global_pos_mean[k] /= num_samples

    if not is_initialized[layer_name]:
        for k in range(max_order + 1):
            stats_dim = global_pos_mean[k].shape[0]
            cdm_models[layer_name][k] = ExponentialFamilyModel(stats_dim=stats_dim).to(device)
            if k > 0:
                cdm_optimizers[layer_name][k] = torch.optim.Adam(
                    cdm_models[layer_name][k].parameters(), lr=lr, weight_decay=0.0
                )
        is_initialized[layer_name] = True

    with torch.no_grad():
        eps = 1e-8
        cdm_models[layer_name][0].theta.copy_(torch.log(global_pos_mean[0] + eps))
        
    return global_pos_mean

def optimize_cdm_order(k, cur_model, prev_model, cur_optimizer, S_all, pos_mean_k, 
                       cd_epochs, current_lr, calc_stats, device, is_init_phase=False):
    """Optimizes the k-th order model (Hierarchical Warm Start is applied only at Epoch 0 when is_init_phase=True)."""
    num_samples = len(S_all)
    cdm_batch_size = 512
    
    for param_group in cur_optimizer.param_groups:
        param_group['lr'] = current_lr
    
    if is_init_phase:
        with torch.no_grad():
            prev_dim = prev_model.theta.shape[0]
            cur_model.theta[:prev_dim].copy_(prev_model.theta)
            
    for inner_epoch in range(cd_epochs):
        indices = np.random.permutation(num_samples)
        for i in range(0, num_samples, cdm_batch_size):
            idx = indices[i:i+cdm_batch_size]
            S_batch = torch.tensor(S_all[idx], dtype=torch.float32).to(device)
            neg_mean = cur_model.compute_negative_stats_mean(S_batch, calc_stats, k)
            cur_model.likelihood_step(cur_optimizer, pos_mean_k, neg_mean)

def optimize_cdm_order_with_tolerance(
    k, current_model, prev_model, optimizer, S_input, global_pos_mean, 
    max_cd_epochs, current_lr, calc_stats, device, is_init_phase=False,
    tolerance=1e-4, min_epochs=10  # Added early stopping threshold and minimum loop count
):
    """
    Optimizes the CDM model for the specified order k (with adaptive termination conditions).
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr

    # Inherit parameters from the (k-1)th order only during Epoch 1 (initialization phase)
    if is_init_phase:
        with torch.no_grad():
            prev_dim = prev_model.theta.shape[0]
            current_model.theta[:prev_dim].copy_(prev_model.theta)

    num_samples = S_input.shape[0]
    cdm_batch_size = 512
    
    for inner_epoch in range(max_cd_epochs):
        indices = np.random.permutation(num_samples)
        max_grad_norm_in_epoch = 0.0  # Record the maximum gradient norm within this epoch
        
        for i in range(0, num_samples, cdm_batch_size):
            idx = indices[i:i+cdm_batch_size]
            S_batch = torch.tensor(S_input[idx], dtype=torch.float32).to(device)

            # Calculate negative sample statistics (model's expected value)
            neg_mean_batch = current_model.compute_negative_stats_mean(
                S_batch=S_batch, 
                calc_stats_fn=calc_stats, 
                k=k
            )

            # Convergence check: calculate the error between data (global_pos_mean) and model (neg_mean_batch)
            with torch.no_grad():
                grad_norm = torch.norm(global_pos_mean - neg_mean_batch, p=2).item()
                if grad_norm > max_grad_norm_in_epoch:
                    max_grad_norm_in_epoch = grad_norm

            # Update parameters
            loss_val = current_model.likelihood_step(
                optimizer=optimizer, 
                pos_mean=global_pos_mean,
                neg_mean=neg_mean_batch
            )
            
        # Evaluate adaptive early stopping
        if inner_epoch >= min_epochs and max_grad_norm_in_epoch < tolerance:
            # Uncomment the print statement below to check the convergence step count for debugging purposes
            # print(f"    [Order {k}] Converged at step {inner_epoch + 1}/{max_cd_epochs} (Grad Norm: {max_grad_norm_in_epoch:.6f})")
            break

def compute_layer_complexity(layer_name, S_all, cdm_models, calc_stats, max_order, device):
    """Calculates exact KL divergence and representational complexity (C) using mini-batches."""
    kl_divergences = {}
    total_complexity_C = 0.0
    
    num_samples = len(S_all)
    batch_size = 512 # split into mini-batches
    
    for k in range(1, max_order + 1):
        kl_sum = 0.0
        for i in range(0, num_samples, batch_size):
            S_batch = torch.tensor(S_all[i:i+batch_size], dtype=torch.float32).to(device)
            kl_val = compute_exact_kl_divergence(
                cdm_models[layer_name][k], cdm_models[layer_name][k-1], S_batch, calc_stats, k
            )
            # compute KL as the weighted sum
            kl_sum += kl_val * (len(S_batch) / num_samples)
            
        kl_divergences[k] = kl_sum
        total_complexity_C += k * kl_sum
        print(f"[{layer_name}] k={k} KL: {kl_sum:.4f}")

    normalization_term = sum(kl_divergences.values())
    epoch_C = total_complexity_C / normalization_term if normalization_term > 0 else 0.0
    print(f"{layer_name} - Representation Complexity C: {epoch_C:.6f}")
    return epoch_C

def track_cdm_drift(layer_name, cdm_models, prev_thetas, run_drift_metrics, max_order):
    """Calculates and records parameter drift of the CDM models (supports arbitrary max_order)."""
    # Get the full parameter vector from the model of max_order
    full_theta_max = cdm_models[layer_name][max_order].theta.detach()
    
    for m in range(1, max_order + 1):
        # Dynamically calculate the indices of the sub-vector corresponding to order m
        start_idx = cdm_models[layer_name][m-1].theta.shape[0]
        end_idx = cdm_models[layer_name][m].theta.shape[0]
        
        # Extract the parameter subset for the corresponding order
        sub_theta = full_theta_max[start_idx:end_idx]
        sub_theta_norm = torch.norm(sub_theta, p=2).item()
        run_drift_metrics['param_norm'][layer_name][m].append(sub_theta_norm)
        
        if prev_thetas[layer_name][m] is not None:
            cos_sim = F.cosine_similarity(sub_theta, prev_thetas[layer_name][m], dim=0).item()
            velocity = torch.norm(sub_theta - prev_thetas[layer_name][m], p=2).item()
            
            run_drift_metrics['cosine_sim'][layer_name][m].append(cos_sim)
            run_drift_metrics['param_velocity'][layer_name][m].append(velocity)
            print(f"Order {m} Parameter Cosine similarity in {layer_name}: {cos_sim:.6f}")
            print(f"Order {m} Parameter Velocity in {layer_name}: {velocity:.6f}")
        else:
            run_drift_metrics['cosine_sim'][layer_name][m].append(1.0)
            run_drift_metrics['param_velocity'][layer_name][m].append(0.0)
            
        prev_thetas[layer_name][m] = sub_theta.clone()

def track_weight_drift(model, layer_name, prev_weights, run_drift_metrics, layer_mapping):
    """Calculates and records the drift of the network weight matrices (supports arbitrary layers/model structures)."""
    current_W = None
    
    # Get the target module name (e.g., 'fc3') from layer_mapping
    module_name = layer_mapping.get(layer_name)
    
    if module_name and hasattr(model, module_name):
        # Access the module dynamically using getattr to retrieve the weights
        target_module = getattr(model, module_name)
        if hasattr(target_module, 'weight') and target_module.weight is not None:
            current_W = target_module.weight.detach().flatten()
            
    if current_W is not None:
        current_W_norm = torch.norm(current_W, p=2).item()
        run_drift_metrics['net_weight_norm'][layer_name].append(current_W_norm)
        
        if prev_weights[layer_name] is not None:
            w_cos_sim = F.cosine_similarity(current_W.unsqueeze(0), prev_weights[layer_name].unsqueeze(0)).item()
            w_velocity = torch.norm(current_W - prev_weights[layer_name], p=2).item()
            print(f"[{layer_name} Network W] cos_sim: {w_cos_sim:.6f}, velocity: {w_velocity:.6f}, norm: {current_W_norm:.6f}")
            
            run_drift_metrics['net_weight_cosine_sim'][layer_name].append(w_cos_sim)
            run_drift_metrics['net_weight_velocity'][layer_name].append(w_velocity)
        else:
            print(f"[{layer_name} Network W] norm: {current_W_norm:.6f}")
            run_drift_metrics['net_weight_cosine_sim'][layer_name].append(1.0)
            run_drift_metrics['net_weight_velocity'][layer_name].append(0.0)
            
        prev_weights[layer_name] = current_W.clone()
    else:
        print(f"Warning: Could not find weight for layer {layer_name} (mapped to {module_name})")