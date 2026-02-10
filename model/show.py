import numpy as np
import matplotlib
matplotlib.use('TkAgg') # <--- Add this line
import matplotlib.pyplot as plt

def load_matrix_from_hex(filename, rows=4, cols=4):
    """
    Parses a hex file into a numpy matrix.
    Assumes the file contains hex strings (e.g., '1A', '0xFF') separated by whitespace or newlines.
    """
    try:
        with open(filename, 'r') as f:
            content = f.read()
            # Split by whitespace and filter out empty strings
            hex_values = content.split()
            
            # Convert hex strings to integers
            # Handling both '0xFF' and 'FF' formats
            int_values = [int(x, 16) for x in hex_values]
            
            # Check if we have enough data
            if len(int_values) < rows * cols:
                print(f"Warning: Not enough data in {filename}. Expected {rows*cols}, got {len(int_values)}.")
                # Pad with zeros if necessary
                int_values += [0] * (rows * cols - len(int_values))
            
            # Truncate if too much data, then reshape
            matrix = np.array(int_values[:rows*cols]).reshape(rows, cols)
            return matrix
            
    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
        return np.zeros((rows, cols))
    except ValueError as e:
        print(f"Error parsing hex in {filename}: {e}")
        return np.zeros((rows, cols))

def plot_matrices(weights, inputs, result):
    """
    Plots the three matrices side-by-side.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    matrices = [
        (weights, 'Weights (4x4)', 'Blues'),
        (inputs, 'Inputs (4x4)', 'Greens'),
        (result, 'NPU Result (4x4)', 'Reds')
    ]
    
    for ax, (data, title, cmap) in zip(axes, matrices):
        # Create heatmap
        im = ax.imshow(data, cmap=cmap)
        ax.set_title(title)
        
        # Annotate each cell with the numeric value
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                # Choose text color based on background intensity for readability
                text_color = "white" if data[i, j] > data.max()/2 else "black"
                ax.text(j, i, str(data[i, j]), 
                        ha="center", va="center", color=text_color, fontweight='bold')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Turn off tick labels for cleaner look
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    # 1. create dummy files for demonstration (You can remove this block)
    # This just ensures the code runs if you copy-paste it immediately.
    with open("weights.hex", "w") as f: f.write("01 02 03 04 05 06 07 08 09 0A 0B 0C 0D 0E 0F 10")
    with open("inputs.hex", "w") as f:  f.write("10 0F 0E 0D 0C 0B 0A 09 08 07 06 05 04 03 02 01")
    with open("result.hex", "w") as f:  f.write("11 11 11 11 11 11 11 11 11 11 11 11 11 11 11 11")

    # 2. Load the data
    # Replace these filenames with your actual .hex file paths
    weights_matrix = load_matrix_from_hex('weights.hex')
    inputs_matrix = load_matrix_from_hex('inputs.hex')
    result_matrix = load_matrix_from_hex('result.hex')

    print("Weights:\n", weights_matrix)
    print("Inputs:\n", inputs_matrix)
    print("Result:\n", result_matrix)

    # 3. Visualize
    plot_matrices(weights_matrix, inputs_matrix, result_matrix)
