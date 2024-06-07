import numpy as np

def _unit_step_func(x):
    return np.where(x>=0, 1, 0)


def get_grid_values(data_df, columns, samples=100):

    x1_values = np.linspace(data_df[columns[0]].min(), data_df[columns[0]].max(), 100)
    x2_values = np.linspace(data_df[columns[1]].min(), data_df[columns[1]].max(), 100)
    x3_values = np.linspace(data_df[columns[2]].min(), data_df[columns[2]].max(), 100)
    x4_values = np.linspace(data_df[columns[3]].min(), data_df[columns[3]].max(), 100)

    x1_grid, x2_grid = np.meshgrid(x1_values, x2_values)
    x3_grid, x4_grid = np.meshgrid(x3_values, x4_values)
    return {'x1': x1_grid, 'x2': x2_grid, 'x3': x3_grid, 'x4':  x4_grid}

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