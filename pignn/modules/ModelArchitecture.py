import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn.inits import reset

# PARAMETERIZABLE NETWORK ARCHITECTURE
# 1. Custom Message Passing Scheme (unchanged)
class MPNNLayer(MessagePassing):
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super(MPNNLayer, self).__init__(aggr="mean")
        self.lin_node = nn.Linear(node_dim, hidden_dim)
        self.lin_edge = nn.Linear(edge_dim, hidden_dim)

        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim))

        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim))
        self.activation = nn.ReLU() 

    def forward(self, x, edge_index, edge_attr):      
        edge_index, edge_attr = add_self_loops(edge_index, edge_attr, num_nodes=x.size(0))
        
        x_transformed = self.lin_node(x)
        edge_attr_transformed = self.lin_edge(edge_attr)

        return self.propagate(edge_index, x=x_transformed, edge_attr=edge_attr_transformed, x_orig=x_transformed)

    def message(self, x_j, edge_attr):
        msg_input = torch.cat([x_j, edge_attr], dim=-1)
        return self.message_mlp(msg_input)

    def update(self, aggr_out, x_orig):
        update_input = torch.cat([aggr_out, x_orig], dim=-1)
        return self.update_mlp(update_input)

# 2. Parameterizable Graph Processing Block
class MPNNBlock(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, num_layers=3):
        super(MPNNBlock, self).__init__()
        
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        # First layer transforms from node_dim to hidden_dim
        self.layers.append(MPNNLayer(node_dim, edge_dim, hidden_dim))
        self.norms.append(nn.LayerNorm(hidden_dim))
        
        # Remaining layers work in hidden_dim
        for i in range(num_layers - 1):
            self.layers.append(MPNNLayer(hidden_dim, edge_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
        
        # Skip connection weights (for layers 2 onwards)
        if num_layers > 1:
            self.skip_weights = nn.Parameter(torch.ones(num_layers - 1))

    def forward(self, x, edge_index, edge_attr):
        # First layer
        x_prev = self.layers[0](x, edge_index, edge_attr)
        x_prev = self.norms[0](x_prev)
        
        # Remaining layers with skip connections
        for i in range(1, self.num_layers):
            x_curr = self.layers[i](x_prev, edge_index, edge_attr)
            x_curr = self.norms[i](x_curr)
            
            # Add skip connection
            x_curr = x_curr + torch.relu(self.skip_weights[i-1]) * x_prev
            x_prev = x_curr
        
        return x_prev

# 3. Parameterizable Fully Connected Block
class FullyConnectedBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, num_layers=4, 
                 normalization='layer'):
        super(FullyConnectedBlock, self).__init__()
        
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.activation = nn.ReLU()
        self.output_activation = nn.Softplus()
        # First layer
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        if normalization == 'batch':
            self.norms.append(nn.BatchNorm1d(hidden_dim))
        elif normalization == 'layer':
            self.norms.append(nn.LayerNorm(hidden_dim))
        
        # Hidden layers
        for i in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            if normalization == 'batch':
                self.norms.append(nn.BatchNorm1d(hidden_dim))
            elif normalization == 'layer':
                self.norms.append(nn.LayerNorm(hidden_dim))
        
        # Output layer (no normalization on final output)
        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        # Forward through all layers except the last
        for i in range(self.num_layers - 1):
            x = self.layers[i](x)
            x = self.norms[i](x)  # Normalize after linear layer
            x = self.activation(x)
        
        # Last layer without normalization/activation
        x = self.layers[-1](x)
        x = self.output_activation(x)
        return x

# 4. Advanced Pooling Strategy
class AdaptivePooling(nn.Module):
    """Learnable combination of different pooling strategies"""
    def __init__(self, hidden_dim, num_pools=3):
        super(AdaptivePooling, self).__init__()
        self.num_pools = num_pools
        self.pool_weights = nn.Parameter(torch.ones(num_pools) / num_pools)
        self.pool_transform = nn.Linear(hidden_dim * num_pools, hidden_dim)
        
    def forward(self, x, batch):
        pools = []
        pools.append(global_mean_pool(x, batch))
        pools.append(global_max_pool(x, batch))
        pools.append(global_add_pool(x, batch))
        
        # Weighted combination
        weighted_pools = [self.pool_weights[i] * pools[i] for i in range(self.num_pools)]
        combined = torch.cat(weighted_pools, dim=1)
        return self.pool_transform(combined)

# 5. Parameterizable Full Model with Advanced Pooling
class VLEAmineCO2(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, output_dim=1, additional_features_dim=3, 
                 graph_layers=3, fc_layers=4, use_adaptive_pooling=False):
        super(VLEAmineCO2, self).__init__()
        
        self.use_adaptive_pooling = use_adaptive_pooling
        self.graph_block = MPNNBlock(node_dim, edge_dim, hidden_dim, num_layers=graph_layers)
        
        if use_adaptive_pooling:
            self.adaptive_pool = AdaptivePooling(hidden_dim)
        
        # Input dimension for FC block is graph embedding + additional features
        fc_input_dim = hidden_dim + additional_features_dim
        self.fc_block = FullyConnectedBlock(fc_input_dim, hidden_dim, output_dim, 
                                          num_layers=fc_layers)
    
    def forward(self, data, extract_embeddings=False):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Graph processing
        x = self.graph_block(x, edge_index, edge_attr)
        
        # Pooling
        if self.use_adaptive_pooling:
            x = self.adaptive_pool(x, batch)
        else:
            x = global_mean_pool(x, batch)
        
        # Extracting embeddings only,
        if extract_embeddings:
            return x
        
        # Combine with additional features
        additional_features = torch.stack([data.conc, data.temp, data.pco2], dim=1).float()
        x = torch.cat([x, additional_features], dim=1)
        
        # Final prediction
        x = self.fc_block(x)
        return x

# 6. Physics-loss definition
def grad_pres(model, data, pco2_std, pco2_mean, s1):
    if s1 == 0:
        return torch.tensor(0.0, device=data.pco2.device, requires_grad=False)
    P = data.pco2.requires_grad_(True)
    alpha_pred = model(data)
    grad_pco2 = torch.autograd.grad(alpha_pred, P, 
                                    grad_outputs=torch.ones_like(alpha_pred),
                                    create_graph=True)[0]
    grad_pco2_real = grad_pco2 * (1/pco2_std)
    penalty = 2 * (F.sigmoid(s1 * F.relu(-grad_pco2_real)) - 0.5)
    return torch.mean(penalty ** 2)

def grad_temp(model, data, temp_std, temp_mean, s2):
    if s2 == 0:
        return torch.tensor(0.0, device=data.temp.device, requires_grad=False)
    T = data.temp.requires_grad_(True)
    alpha_pred = model(data)
    grad_temp = torch.autograd.grad(alpha_pred, T, 
                                    grad_outputs=torch.ones_like(alpha_pred),
                                    create_graph=True)[0]
    grad_temp_real = grad_temp * (1/temp_std)
    penalty = 2 * (F.sigmoid(s2 * F.relu(grad_temp_real)) - 0.5)
    return torch.mean(penalty ** 2)

def grad_conc(model, data, conc_std, conc_mean, s3):
    if s3 == 0:
        return torch.tensor(0.0, device=data.conc.device, requires_grad=False)
    C = data.conc.requires_grad_(True)
    alpha_pred = model(data)
    grad_conc = torch.autograd.grad(alpha_pred, C, 
                                    grad_outputs=torch.ones_like(alpha_pred),
                                    create_graph=True)[0]
    grad_conc_real =  grad_conc * (1/conc_std)
    penalty = 2 * (F.sigmoid(s3 * F.relu(grad_conc_real)) - 0.5)
    return torch.mean(penalty ** 2)

# 7. Mean Squared Logarithmic Error Loss Module
class MSLELoss(nn.Module):
    """Mean Squared Logarithmic Error Loss
    
    MSLE = mean((log(y_true + 1) - log(y_pred + 1))²)
    
    Args:
        epsilon (float): Small value added to prevent log(0). Default: 1.0
        reduction (str): 'mean', 'sum', or 'none'. Default: 'mean'
    """
    def __init__(self, epsilon=1.0, reduction='mean'):
        super(MSLELoss, self).__init__()
        self.epsilon = epsilon
        self.reduction = reduction
        
    def forward(self, y_pred, y_true):
        # Add epsilon to prevent log(0)
        log_pred = torch.log(y_pred + self.epsilon)
        log_true = torch.log(y_true + self.epsilon)
        
        # Compute squared differences
        squared_log_diff = (log_true - log_pred) ** 2
        
        # Apply reduction
        if self.reduction == 'mean':
            return torch.mean(squared_log_diff)
        elif self.reduction == 'sum':
            return torch.sum(squared_log_diff)
        elif self.reduction == 'none':
            return squared_log_diff
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}")