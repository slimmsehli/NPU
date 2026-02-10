import os
import sys

# Import the NPU classes from the other file
from npu import NPUConfig, Memory, SystolicArray, PPU, Controller

class FileUtils:
    @staticmethod
    def read_hex_file(filename):
        """
        Reads a HEX file (whitespace separated) and returns a string of tokens.
        Compatible with Verilog $readmemh format.
        """
        if not os.path.exists(filename):
            print(f"Warning: {filename} not found. Returning empty string.")
            return ""
        
        with open(filename, 'r') as f:
            content = f.read()
            tokens = content.split()
        
        # Join them back into a single string for the Memory.load_hex_string method
        return " ".join(tokens)

    @staticmethod
    def write_hex_file(filename, hex_string):
        """
        Writes the output HEX string to a file, formatted nicely (16 bytes per line).
        """
        tokens = hex_string.split()
        with open(filename, 'w') as f:
            for i, token in enumerate(tokens):
                f.write(token + " ")
                if (i + 1) % 16 == 0: # Newline every 16 bytes for readability
                    f.write("\n")
        print(f"[FileUtils] Wrote output to {filename}")

# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    # 1. Configuration
    array_size = 4
    config = NPUConfig(array_size=array_size, data_width=8, mem_depth=1024)
    mem = Memory(config)
    array = SystolicArray(config)
    ppu = PPU(config)
    ctrl = Controller(mem, array, ppu, config)

    # 2. File Paths
    weight_file = "./hex/weights.hex"
    input_file = "./hex/inputs.hex"
    output_file = "./hex/result.hex"

    # 3. Check for existence of input files
    if not os.path.exists(weight_file) or not os.path.exists(input_file):
        print(f"Error: Please ensure {weight_file} and {input_file} exist.")
        # Optional: Create dummy files if needed (Uncomment below to auto-generate)
        # with open(weight_file, "w") as f: f.write("01 " * 16)
        # with open(input_file, "w") as f: f.write("02 " * 16)
        # sys.exit(1)

    # 4. Load Data from Files
    print(f"Loading Weights from {weight_file}...")
    w_data_str = FileUtils.read_hex_file(weight_file)
    if w_data_str:
        mem.load_hex_string(start_addr=0x000, hex_string=w_data_str)

    print(f"Loading Inputs from {input_file}...")
    i_data_str = FileUtils.read_hex_file(input_file)
    if i_data_str:
        mem.load_hex_string(start_addr=0x100, hex_string=i_data_str)

    # 5. Define Program / Instruction Stream
    # We assume Weights are at 0x000, Inputs at 0x100, Result goes to 0x200
    program = [
        {'op': 'LOAD_WEIGHTS', 'addr': 0x000},
        {'op': 'MATMUL',       'src': 0x100, 'dst': 0x200},
        {'op': 'HALT'}
    ]
    
    # 6. Run Simulation
    ctrl.execute_program(program)

    # 7. Export Results
    # We read 16 bytes (4x4 matrix) from the destination address
    result_hex_str = mem.dump_hex(start_addr=0x200, size=array_size*array_size)
    
    FileUtils.write_hex_file(output_file, result_hex_str)
    
    print("\n--- Test Complete ---")
    print(f"Result Content: {result_hex_str}")
