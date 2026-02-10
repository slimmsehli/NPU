import os
import sys

# Import the Wrapper (Driver) instead of raw components
# Ensure npu_wrapper.py is in the same folder
from npu_wrapper import NPUDriver

class FileUtils:
    @staticmethod
    def read_hex_file_to_list(filename):
        """
        Reads a HEX file and returns a LIST of integers.
        """
        if not os.path.exists(filename):
            print(f"Warning: {filename} not found.")
            return []
        
        with open(filename, 'r') as f:
            content = f.read()
            tokens = content.split()
            
        # Convert hex tokens to integers
        try:
            return [int(t, 16) for t in tokens]
        except ValueError as e:
            print(f"Error parsing hex in {filename}: {e}")
            return []

    @staticmethod
    def write_hex_file(filename, data_list):
        """
        Writes a list of integers to a HEX file.
        """
        with open(filename, 'w') as f:
            for i, val in enumerate(data_list):
                f.write(f"{val:02X} ")
                if (i + 1) % 16 == 0: 
                    f.write("\n")
        print(f"[FileUtils] Wrote output to {filename}")

# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    # 1. Initialize the Driver
    # The driver automatically sets up Memory, Array, PPU, and Controller
    driver = NPUDriver()
    
    # 2. File Paths
    # We now look for separate weight files for each layer
    input_file    = "./hex/inputs.hex"
    w_layer1_file = "./hex/weights_L1.hex"
    w_layer2_file = "./hex/weights_L2.hex"
    output_file   = "./hex/final_result.hex"

    # 3. Load Data from Files
    print("--- Loading Files ---")
    input_data = FileUtils.read_hex_file_to_list(input_file)
    w_L1_data  = FileUtils.read_hex_file_to_list(w_layer1_file)
    w_L2_data  = FileUtils.read_hex_file_to_list(w_layer2_file)

    # Validation: Ensure we actually loaded data
    if not input_data:
        print(f"Error: {input_file} is missing or empty.")
        sys.exit(1)
    if not w_L1_data:
        print(f"Error: {w_layer1_file} is missing or empty.")
        sys.exit(1)
    if not w_L2_data:
        print(f"Error: {w_layer2_file} is missing or empty.")
        sys.exit(1)

    # 4. Define the Network Architecture
    # We create a list of dictionaries. Each dict represents one layer.
    # The driver will iterate through this list.
    network_layers = [
        {
            # Layer 1
            'weights': w_L1_data, 
            'scale': 1.0,        # Adjust scaling if L1 output is too large
            'zero_point': 0
        },
        {
            # Layer 2
            'weights': w_L2_data, 
            'scale': 1.0,        # Adjust scaling for final output
            'zero_point': 0
        }
    ]

    # 5. Run Inference
    # The driver handles memory swapping (ping-pong), loading weights,
    # and generating instructions for every layer automatically.
    final_result = driver.run_inference(input_data, network_layers)

    # 6. Export Final Result
    FileUtils.write_hex_file(output_file, final_result)
    
    print("\n--- Test Complete ---")
    print(f"Final Result (First 16 bytes): {[hex(x) for x in final_result]}")
    print(f"Check {output_file} for full output.")
