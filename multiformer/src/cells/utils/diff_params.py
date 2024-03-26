import torch
import torch.nn as nn
import pandas as pd

def compare_model_architectures(model1, model2=None, include_layer_type=False):
    """
    Compare the architectures of two PyTorch models.

    Args:
    - model1: First PyTorch model.
    - model2: Second PyTorch model.
    """
    if not model2:
        model2 = nn.Module()
    # Initialize lists to store architecture details
    model1_architecture = []
    model2_architecture = []

    # Iterate through the layers/modules of each model
    for name, module in model1.named_modules():
        layer_info = {
            'Layer Name': name,
            'Parameters Shape': [tuple(param.shape) for param in module.parameters()],
            'Input Shape': module.in_features if hasattr(module, 'in_features') else None,
            'Output Shape': module.out_features if hasattr(module, 'out_features') else None
        }
        layer_info['Layer Type'] = str(module.__class__.__name__)
        model1_architecture.append(layer_info)

    for name, module in model2.named_modules():
        layer_info = {
            'Layer Name': name,
            'Parameters Shape': [tuple(param.shape) for param in module.parameters()],
            'Input Shape': module.in_features if hasattr(module, 'in_features') else None,
            'Output Shape': module.out_features if hasattr(module, 'out_features') else None
        }
        layer_info['Layer Type'] = str(module.__class__.__name__)
        model2_architecture.append(layer_info)

    # Create DataFrame for each model's architecture
    df_model1 = pd.DataFrame(model1_architecture)
    df_model2 = pd.DataFrame(model2_architecture)

    # Add column to indicate model
    df_model1['Model'] = 'Model 1'
    df_model2['Model'] = 'Model 2'

    # Merge DataFrames on layer name
    df_comparison = pd.merge(df_model1, df_model2, on='Layer Name', how='outer', suffixes=('_Model1', '_Model2'))

    # Reorder columns
    columns_order = ['Layer Name']

    columns_order.append('Layer Type_Model1')
    columns_order.append('Layer Type_Model2')
    columns_order.extend(['Parameters Shape_Model1', 'Parameters Shape_Model2',
                     'Input Shape_Model1', 'Input Shape_Model2', 'Output Shape_Model1', 'Output Shape_Model2'])
    df_comparison = df_comparison[columns_order]

    # Display comparison in table format
    return df_comparison

if __name__=="__main__":
    compare_model_architectures(nn.Module())
