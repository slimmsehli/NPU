# ... (Import classes and FileUtils as before) ...
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

if __name__ == "__main__":
    # 1. Configuration
    config = NPUConfig(array_size=4, data_width=8, mem_depth=1024)
    mem = Memory(config)
    array = SystolicArray(config)
    ppu = PPU(config)
    ctrl = Controller(mem, array, ppu, config)

    # 2. File Paths (Ensure these exist or use dummy data)
    weight_file = "./hex/weights.hex"
    input_file = "./hex/inputs.hex"
    output_file = "./hex/result.hex"
    
    # ... (Load files as before) ...
    # Assumes you have loaded data into memory already using FileUtils

    # 5. Define Program with SMART QUANTIZATION
    # We are adding 'scale': 10.0. 
    # This means the huge accumulated result will be divided by 10 before saving.
    program = [
        {'op': 'LOAD_WEIGHTS', 'addr': 0x000},
        
        # Here is the smart part:
        {'op': 'MATMUL',       'src': 0x100, 'dst': 0x200, 'scale': 1.0, 'zero_point': 0},
        
        {'op': 'HALT'}
    ]
    
    # 6. Run Simulation
    ctrl.execute_program(program)

    # 7. Export Results
    result_hex_str = mem.dump_hex(start_addr=0x200, size=16)
    FileUtils.write_hex_file(output_file, result_hex_str)
    
    print("\n--- Test Complete ---")
    print(f"Result Content: {result_hex_str}")
