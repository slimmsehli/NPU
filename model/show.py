import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Use TkAgg if on a windowed system, otherwise it might fail on some setups
try:
    import matplotlib
    matplotlib.use('TkAgg') 
except:
    pass

def load_matrix_from_hex(filename, rows=4, cols=4):
    """
    Parses a hex file into a numpy matrix.
    Handles both 8-bit (2 chars) and 32-bit (8 chars) hex values.
    """
    try:
        with open(filename, 'r') as f:
            content = f.read()
            # Split by whitespace
            tokens = content.split()
            
            # Convert hex strings to integers
            int_values = []
            for t in tokens:
                try:
                    int_values.append(int(t, 16))
                except ValueError:
                    continue # Skip non-hex tokens
            
            # Pad with zeros if not enough data
            if len(int_values) < rows * cols:
                print(f"Warning: {filename} has {len(int_values)} values, expected {rows*cols}. Padding with 0.")
                int_values += [0] * (rows * cols - len(int_values))
            
            # Create matrix
            matrix = np.array(int_values[:rows*cols]).reshape(rows, cols)
            return matrix
            
    except FileNotFoundError:
        print(f"Error: File {filename} not found. Returning zeros.")
        return np.zeros((rows, cols))

def plot_full_pipeline(weights, inputs, raw, activated, final):
    """
    Plots the 5 stages of the NPU pipeline.
    """
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    
    # Define the 5 plots
    plots = [
        (weights,   '1. Weights (8-bit)',   'Blues'),
        (inputs,    '2. Inputs (8-bit)',    'Greens'),
        (raw,       '3. Raw Accum (32-bit)', 'Oranges'), # Raw output can be huge
        (activated, '4. Activated (ReLU)',  'Purples'), # No negatives
        (final,     '5. Final Output (8-bit)','Reds')    # Quantized back to 0-255
    ]
    
    for ax, (data, title, cmap) in zip(axes, plots):
        # Determine normalization for color scaling
        # For 8-bit, we stick to 0-255. For 32-bit, we use min/max of the data.
        if '32-bit' in title or 'Activated' in title:
            norm = mcolors.Normalize(vmin=data.min(), vmax=data.max())
        else:
            norm = mcolors.Normalize(vmin=0, vmax=255)

        im = ax.imshow(data, cmap=cmap, norm=norm)
        ax.set_title(title, fontsize=10, fontweight='bold')
        
        # Annotate cells
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i, j]
                
                # Dynamic text color (white if dark background, black if light)
                # Calculate relative brightness of the cell
                cell_intensity = (val - norm.vmin) / (norm.vmax - norm.vmin + 1e-5)
                text_color = "white" if cell_intensity > 0.5 else "black"
                
                # For large numbers, use smaller font
                font_size = 8 if val > 999 else 10
                
                ax.text(j, i, str(val), 
                        ha="center", va="center", 
                        color=text_color, fontweight='bold', fontsize=font_size)
        
        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    array_size = 8
    # 1. Load the 5 matrices
    # Ensure these filenames match what your NPU script generates!
    w_mat = load_matrix_from_hex('./hex/weights.hex', array_size, array_size)
    i_mat = load_matrix_from_hex('./hex/inputs.hex', array_size, array_size)
    
    # Intermediate dumps (Make sure you enabled these in your Controller)
    raw_mat = load_matrix_from_hex('./hex/result_multiplication.hex', array_size, array_size)
    act_mat = load_matrix_from_hex('./hex/result_activated.hex', array_size, array_size)
    fin_mat = load_matrix_from_hex('./hex/result_final.hex', array_size, array_size)

    # 2. Visualize
    print("Visualizing NPU Pipeline...")
    plot_full_pipeline(w_mat, i_mat, raw_mat, act_mat, fin_mat)



