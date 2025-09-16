import torch
import re
from ModelArchitecture import VLEAmineCO2

def load_model_smart(model_path, device='cpu'):
    """
    Automatically load model with architecture parameters inferred from saved weights.
    
    Args:
        model_path (str): Path to the .pth file
        device (str): Device to load the model on
    
    Returns:
        tuple: (model, config_dict)
    """
    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Check if config is saved (Approach 1)
    if isinstance(checkpoint, dict) and 'config' in checkpoint:
        config = checkpoint['config']
        state_dict = checkpoint['model_state_dict']
        print("Found saved config in checkpoint")
    else:
        # Auto-detect from state dict (Approach 2)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        config = infer_architecture_from_state_dict(state_dict)
        print("Architecture inferred from state dict")
    
    # Create model with inferred config
    model = VLEAmineCO2(
        node_dim=config['node_dim'],
        edge_dim=config['edge_dim'], 
        hidden_dim=config['hidden_dim'],
        output_dim=config['output_dim'],
        additional_features_dim=config['additional_features_dim'],
        graph_layers=config['graph_layers'],
        fc_layers=config['fc_layers'],
        use_adaptive_pooling=config['use_adaptive_pooling']
    )
    
    # Load the weights
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully with config: {config}")
    return model, config

def infer_architecture_from_state_dict(state_dict):
    """
    Infer architecture parameters from the state dictionary.
    """
    config = {}
    
    # Extract dimensions from first graph layer
    first_node_layer = 'graph_block.layers.0.lin_node.weight'
    first_edge_layer = 'graph_block.layers.0.lin_edge.weight'
    
    if first_node_layer in state_dict:
        # lin_node transforms node_dim -> hidden_dim
        hidden_dim, node_dim = state_dict[first_node_layer].shape
        config['node_dim'] = node_dim
        config['hidden_dim'] = hidden_dim
    else:
        # Fallback defaults
        config['node_dim'] = 9
        config['hidden_dim'] = 64
    
    if first_edge_layer in state_dict:
        # lin_edge transforms edge_dim -> hidden_dim  
        _, edge_dim = state_dict[first_edge_layer].shape
        config['edge_dim'] = edge_dim
    else:
        config['edge_dim'] = 3
    
    # Count graph layers by looking at the MPNN layers
    graph_layer_pattern = r'graph_block\.layers\.(\d+)\.'
    graph_layer_indices = set()
    for key in state_dict.keys():
        match = re.search(graph_layer_pattern, key)
        if match:
            graph_layer_indices.add(int(match.group(1)))
    config['graph_layers'] = max(graph_layer_indices) + 1 if graph_layer_indices else 3
    
    # Count FC layers by looking at the fully connected block
    fc_layer_pattern = r'fc_block\.layers\.(\d+)\.'
    fc_layer_indices = set()
    for key in state_dict.keys():
        match = re.search(fc_layer_pattern, key)
        if match:
            fc_layer_indices.add(int(match.group(1)))
    config['fc_layers'] = max(fc_layer_indices) + 1 if fc_layer_indices else 4
    
    # Get output dimension from last FC layer
    if fc_layer_indices:
        last_fc_layer = f'fc_block.layers.{max(fc_layer_indices)}.weight'
        if last_fc_layer in state_dict:
            output_dim, _ = state_dict[last_fc_layer].shape
            config['output_dim'] = output_dim
        else:
            config['output_dim'] = 1
    else:
        config['output_dim'] = 1
    
    # Check for adaptive pooling
    config['use_adaptive_pooling'] = any(key.startswith('adaptive_pool.') for key in state_dict.keys())
    
    # Additional features dimension - check the FC block input dimension
    if fc_layer_indices:
        first_fc_layer = f'fc_block.layers.0.weight'
        if first_fc_layer in state_dict:
            _, fc_input_dim = state_dict[first_fc_layer].shape
            # FC input = hidden_dim + additional_features_dim
            config['additional_features_dim'] = fc_input_dim - config['hidden_dim']
        else:
            config['additional_features_dim'] = 3
    else:
        config['additional_features_dim'] = 3
    
    # Ensure additional_features_dim is positive and reasonable
    if config['additional_features_dim'] <= 0:
        config['additional_features_dim'] = 3
    
    return config

def load_model_for_inference(model_path, device='cpu'):
    """
    Convenience function for inference - just returns the model ready to use.
    """
    model, _ = load_model_smart(model_path, device)
    return model

def save_model_with_config(model, optimizer, epoch, loss, config, filepath):
    """
    Save model with configuration for easy reloading.
    
    Args:
        model: The trained model
        optimizer: The optimizer (optional)
        epoch: Current epoch
        loss: Current loss
        config: Dictionary containing model configuration
        filepath: Path to save the model
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'loss': loss,
        'config': config
    }, filepath)
    print(f"Model saved with config to {filepath}")

# EXECUTE CALL
if __name__ == "__main__":
    # Method 1: Direct loading for inference
    model = load_model_for_inference('path/to/model.pth')
    
    # Method 2: With config info
    model, config = load_model_smart('path/to/model.pth')
    print(f"Loaded model with config: {config}")
    
    # Example of how to save a model with config during training
    # save_model_with_config(model, optimizer, epoch, loss, config, 'model_with_config.pth')