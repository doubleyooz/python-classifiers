
import numpy as np

def _sigmoid_derivative(self, x):
    return x * (1.0 - x)

def _sigmoid(self, x):       
    return 1.0/(1.0 + np.exp(-x))

def _unit_step_func(x):
    return np.where(x>=0, 1, 0)

def re_lu(Z):
    return np.maximum(0, Z)

def softmax(Z):
    return np.exp(Z) / np.sum(np.exp(Z))

def swap_zero_one_explicit(num):
    if num == 0:
        return 1
    elif num == 1:
        return 0
    else:
        raise ValueError("Input must be 0 or 1")
    
       
def get_grid_values(data_df, columns, samples=100):
    # Initialize an empty dictionary to store the grids
    grids = {}

    values = []
  
    
    # Iterate over the specified columns
    for col in columns:      
        # Generate linearly spaced values for the current column
        values.append(np.linspace(data_df[col].min(), data_df[col].max(), samples))

    for i, value in enumerate(values):
        print(i, len(values))
        
        if i % 2 != 0:
            continue

        # Create a meshgrid for the current column
        col_grid1, col_grid2 = np.meshgrid(value, values[i+1])

        # Store the grid in the dictionary
        grids[f'x{i + 1}'] = col_grid1
        grids[f'x{i + 2}'] = col_grid2
    
    if len(values) % 2 != 0:
        values_length = len(values) - 1
        col_grid1, col_grid2 = np.meshgrid(values[values_length], values[values_length])

        grids[f'x{values_length + 1}'] = col_grid1
        grids[f'x{values_length + 2}'] = col_grid2
    
    
    return grids