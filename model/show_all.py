import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import glob
import os
import math

# Force TkAgg backend for window display compatibility
try:
    import matplotlib
    matplotlib.use('TkAgg') 
except:
    pass

def load_matrix_from_hex(filename, rows=8, cols=8):
    """
    Parses a hex file into a numpy matrix.
    """
    try:
        with open(filename, 'r') as f:
            content = f.read()
            tokens = content.split()
            
            int_values = []
            for t in tokens:
                try:
                    int_values.append(int(t, 16))
                except ValueError:
                    continue 
            
            # Pad with zeros if data is missing
            target_size = rows * cols
            if len(int_values) < target_size:
                int_values += [0] * (target_size - len(int_values))
            
            # Truncate if too much data (take first 16 values)
            matrix = np.array(int_values[:target_size]).reshape(rows, cols)
            return matrix
            
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return np.zeros((rows, cols))

def plot_all_hex_files(folder_path='./hex', rows=8, cols=8):
    """
    Finds all .hex files in the folder and plots them dynamically.
    """
    # 1. Find files
    # We look for all .hex files. 
    # specific_order ensures inputs/weights come before layers if named consistently
    files = sorted(glob.glob(os.path.join(folder_path, "*.hex")))
    
    if not files:
        print(f"No .hex files found in '{folder_path}'. checking current directory...")
        files = sorted(glob.glob("*.hex"))
        if not files:
            print("No hex files found anywhere!")
            return

    num_files = len(files)
    print(f"Found {num_files} files to plot.")

    # 2. Calculate Grid Dimensions (approx square)
    cols = 3 # Fixed width of 3 is usually good for readability
    rows = math.ceil(num_files / cols)
    
    # 3. Setup Plot
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3 * rows))
    axes = np.array(axes).flatten() # Flatten to easy 1D iteration

    # 4. Iterate and Plot
    for i, filepath in enumerate(files):
        ax = axes[i]
        filename = os.path.basename(filepath)
        
        # Load Data
        data = load_matrix_from_hex(filepath, rows=8, cols=8)
        
        # Determine Color Map
        # Green for Inputs, Blue for Weights, Orange for Raw (large), Red for Output
        if "input" in filename.lower(): cmap = "Greens"
        elif "weight" in filename.lower(): cmap = "Blues"
        elif "raw" in filename.lower(): cmap = "Oranges"
        elif "final" in filename.lower() or "result" in filename.lower(): cmap = "Reds"
        elif "activ" in filename.lower(): cmap = "Purples"
        else: cmap = "Greys"

        # Normalization
        # If max value is > 255, it's likely a 32-bit raw accumulator. 
        # We normalize strictly to the data range so colors show contrast.
        # If it's 8-bit, we normalize 0-255 for consistency.
        if data.max() > 255:
            norm = mcolors.Normalize(vmin=data.min(), vmax=data.max())
            data_type_label = " (32-bit)"
        else:
            norm = mcolors.Normalize(vmin=0, vmax=255)
            data_type_label = " (8-bit)"

        im = ax.imshow(data, cmap=cmap, norm=norm)
        ax.set_title(filename + data_type_label, fontsize=10, fontweight='bold')
        
        # Annotate Values
        for r in range(data.shape[0]):
            for c in range(data.shape[1]):
                val = data[r, c]
                
                # Dynamic text color logic
                # If cell is dark, text is white. If cell is light, text is black.
                if data.max() == data.min(): # Avoid div by zero
                    intensity = 0.5
                else:
                    intensity = (val - data.min()) / (data.max() - data.min())
                
                text_color = "white" if intensity > 0.5 else "black"
                
                # Small font for huge numbers
                font_size = 8 if val > 999 else 10
                
                ax.text(c, r, str(val), 
                        ha="center", va="center", 
                        color=text_color, fontweight='bold', fontsize=font_size)
        
        # Hide ticks
        ax.set_xticks([])
        ax.set_yticks([])

    # 5. Hide Empty Subplots (if any)
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Change this to wherever your hex files are stored
    # If they are in the same folder as the script, use '.'
    plot_all_hex_files('./hex', rows=8, cols=8)
