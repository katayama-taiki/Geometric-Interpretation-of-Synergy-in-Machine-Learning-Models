import matplotlib.pyplot as plt
import numpy as np
import os

def plot_complexity_dynamics(final_results, epochs, calc_interval, target_layers, save_path=None):
    """
    Plots the representational complexity (C) across specified layers and test accuracy over epochs.
    Automatically handles any number of target layers and dynamically assigns colors.
    """
    num_runs = len(final_results['accuracy'])
    epochs_x = [1] + [e + 1 for e in range(calc_interval - 1, epochs, calc_interval)]
    
    def calc_mean_ci(data):
        """Calculate mean and 95% confidence interval across runs."""
        mean = np.mean(data, axis=0)
        ci = 1.96 * (np.std(data, axis=0) / np.sqrt(num_runs))
        return mean, ci
        
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Use Matplotlib's 'tab10' colormap for dynamic color assignment
    cmap = plt.get_cmap('tab10')
    
    # Plot Complexity for each target layer dynamically
    for idx, layer in enumerate(target_layers):
        if layer in final_results and len(final_results[layer]) > 0:
            layer_data = np.array(final_results[layer])
            mean_val, ci_val = calc_mean_ci(layer_data)
            
            # Dynamically pull colors based on layer index
            color = cmap(idx % 10)
            
            ax1.plot(epochs_x, mean_val, color=color, label=layer, linewidth=2)
            ax1.fill_between(epochs_x, mean_val - ci_val, mean_val + ci_val, color=color, alpha=0.25)
            
    ax1.set_xlabel('Training Epoch', fontsize=14)
    ax1.set_ylabel('Representational Complexity $C$', fontsize=14)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Plot Test Accuracy (Secondary Y-Axis)
    ax2 = ax1.twinx()
    color_acc = 'gray'
    acc_data = np.array(final_results['accuracy'])
    mean_acc, ci_acc = calc_mean_ci(acc_data)
    
    ax2.plot(epochs_x, mean_acc, color=color_acc, label='Test Accuracy', linewidth=2, linestyle='--')
    ax2.fill_between(epochs_x, mean_acc - ci_acc, mean_acc + ci_acc, color=color_acc, alpha=0.15)
    
    ax2.set_ylabel('Test Accuracy (%)', fontsize=14, color=color_acc)
    ax2.tick_params(axis='y', labelcolor=color_acc)
    ax2.set_ylim(0, 105)
    
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right', fontsize=12, framealpha=0.8)
    
    plt.title('Learning Dynamics: Representational Complexity through Layers', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    plt.show()

def plot_drift_dynamics(final_results, epochs, calc_interval, max_order, target_layers, save_dir='.'):
    """
    Plots the dynamics measures (cosine similarity, velocity, norm) for both 
    CDM representation parameters and network weights for each target layer.
    """
    num_runs = len(final_results['accuracy'])
    epochs_x = [1] + [e + 1 for e in range(calc_interval - 1, epochs, calc_interval)]
    
    def get_drift_stats(metric_name, layer, target_k=None):
        """Helper to extract and average drift statistics across runs."""
        extracted_data = []
        for run_idx in range(num_runs):
            run_data = final_results['drift'][metric_name][run_idx][layer]
            
            if 'net_weight' in metric_name:
                extracted_data.append(run_data)
            else:
                if target_k is not None:
                    extracted_data.append(run_data[target_k])
                else:
                    k_arrays = [run_data[k] for k in range(1, max_order + 1)]
                    extracted_data.append(np.mean(k_arrays, axis=0))
                    
        data_array = np.array(extracted_data)
        mean = np.mean(data_array, axis=0)
        ci = 1.96 * (np.std(data_array, axis=0) / np.sqrt(num_runs))
        return mean, ci

    # Use Matplotlib's 'tab10' colormap dynamically for CDM orders
    cmap = plt.get_cmap('tab10')
    w_color = '#555555' # Color for network weights (Control)
    uniform_lw = 2.0  
    
    for layer in target_layers:
        fig, axes = plt.subplots(1, 4, figsize=(24, 5))
        
        # ----------------------------------------------------
        # [Panel 1] Theta Cosine Similarity
        # ----------------------------------------------------
        ax = axes[0]
        for k in range(1, max_order + 1):
            mean_val, ci_val = get_drift_stats('cosine_sim', layer, target_k=k)
            c = cmap((k - 1) % 10)
            ax.plot(epochs_x, mean_val, color=c, label=f'Order $k={k}$', linewidth=uniform_lw)
            ax.fill_between(epochs_x, mean_val - ci_val, mean_val + ci_val, color=c, alpha=0.15)
        
        ax.set_title(rf'$\theta$ Cosine Similarity in {layer}', fontsize=13)
        ax.set_xlabel('Training Epoch', fontsize=12)
        ax.set_ylabel(r'$\cos(\theta^{(t)}, \theta^{(t-1)})$', fontsize=12)
        ax.set_ylim(top=1.01)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='lower right', fontsize=10)

        # ----------------------------------------------------
        # [Panel 2] Theta Parameter Velocity
        # ----------------------------------------------------
        ax = axes[1]
        for k in range(1, max_order + 1):
            mean_val, ci_val = get_drift_stats('param_velocity', layer, target_k=k)
            c = cmap((k - 1) % 10)
            ax.plot(epochs_x, mean_val, color=c, label=f'Order $k={k}$', linewidth=uniform_lw)
            ax.fill_between(epochs_x, mean_val - ci_val, mean_val + ci_val, color=c, alpha=0.15)
            
        ax.set_title(rf'$\theta$ Velocity in {layer}', fontsize=13)
        ax.set_xlabel('Training Epoch', fontsize=12)
        ax.set_ylabel(r'$||\theta^{(t)} - \theta^{(t-1)}||_2$', fontsize=12)
        ax.set_ylim(bottom=0.0)
        ax.grid(True, linestyle='--', alpha=0.6)

        # ----------------------------------------------------
        # [Panel 3] Theta L2 Norm
        # ----------------------------------------------------
        ax = axes[2]
        for k in range(1, max_order + 1):
            mean_val, ci_val = get_drift_stats('param_norm', layer, target_k=k)
            c = cmap((k - 1) % 10)
            ax.plot(epochs_x, mean_val, color=c, label=f'Order $k={k}$', linewidth=uniform_lw)
            ax.fill_between(epochs_x, mean_val - ci_val, mean_val + ci_val, color=c, alpha=0.15)
            
        ax.set_title(rf'$\theta$ Absolute Norm in {layer}', fontsize=13)
        ax.set_xlabel('Training Epoch', fontsize=12)
        ax.set_ylabel(r'$||\theta^{(t)}||_2$', fontsize=12)
        ax.set_ylim(bottom=0.0)
        ax.grid(True, linestyle='--', alpha=0.6)

        # ----------------------------------------------------
        # [Panel 4] Network Weight Cosine Similarity (Control)
        # ----------------------------------------------------
        ax = axes[3]
        mean_val, ci_val = get_drift_stats('net_weight_cosine_sim', layer)
        ax.plot(epochs_x, mean_val, color=w_color, label=f'{layer} Weights', linewidth=2.5, linestyle='--')
        ax.fill_between(epochs_x, mean_val - ci_val, mean_val + ci_val, color=w_color, alpha=0.15)
        
        ax.set_title(rf'$W$ Cosine Similarity in {layer}', fontsize=13)
        ax.set_xlabel('Training Epoch', fontsize=12)
        ax.set_ylabel(r'$\cos(W^{(t)}, W^{(t-1)})$', fontsize=12)
        ax.set_ylim(top=1.01)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='lower right', fontsize=10)

        plt.suptitle(f'[{layer}] Representation Order Dynamics vs. Network Weights', fontsize=15, y=1.05)
        plt.tight_layout()
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'dynamics_decomposition_{layer}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()