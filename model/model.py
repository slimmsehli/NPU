import math

class NPUConfig:
    """
    Configuration for the NPU Architecture.
    """
    def __init__(self, array_size=4, data_width=8, mem_depth=1024):
        self.array_size = array_size  # N x N Systolic Array
        self.data_width = data_width  # Bit width (e.g., 8-bit integer)
        self.mem_depth = mem_depth    # Number of addresses in unified memory
        self.max_val = (2**data_width) - 1

class Memory:
    """
    Models the Unified Buffer/SRAM.
    Stores data as integers, but handles HEX I/O.
    """
    def __init__(self, config):
        self.mem = [0] * config.mem_depth
        self.config = config

    def load_hex_string(self, start_addr, hex_string):
        """Loads a space-separated HEX string into memory."""
        tokens = hex_string.strip().split()
        for i, token in enumerate(tokens):
            if start_addr + i < len(self.mem):
                self.mem[start_addr + i] = int(token, 16)
            else:
                raise ValueError("Memory Overflow")
        print(f"[Memory] Loaded {len(tokens)} bytes at Address {start_addr}")

    def read_block(self, addr, size):
        """Reads a block of data for the Systolic Array."""
        return self.mem[addr : addr + size]

    def write_block(self, addr, data):
        """Writes data back to memory (e.g., from PPU)."""
        for i, val in enumerate(data):
            # Clamp to max value to simulate overflow behavior of hardware
            clamped_val = min(max(val, 0), 2**32) # Assuming 32-bit accumulators
            self.mem[addr + i] = clamped_val

    def dump_hex(self, start_addr, size):
        """Returns memory content as HEX string."""
        chunk = self.mem[start_addr : start_addr + size]
        return ' '.join([f'{x:02X}' for x in chunk])

class SystolicArray:
    """
    Models a Weight-Stationary Systolic Array.
    Performs Matrix Multiplication (A x B).
    """
    def __init__(self, config):
        self.n = config.array_size
        self.config = config
        self.weights = [[0]*self.n for _ in range(self.n)] # Internal Registers
    
    def load_weights(self, weight_matrix_flat):
        """
        Loads weights into the array (Stationary Phase).
        Expects a flat list of N*N elements (Row-Major).
        """
        if len(weight_matrix_flat) != self.n * self.n:
            raise ValueError(f"Weight mismatch: Expected {self.n*self.n}, got {len(weight_matrix_flat)}")
        
        print("[Systolic] Loading Weights...")
        for r in range(self.n):
            for c in range(self.n):
                self.weights[r][c] = weight_matrix_flat[r * self.n + c]

    def run_matmul(self, input_matrix_flat):
        """
        Simulates the execution phase.
        Input A (NxN) x Weights B (NxN) = Output C (NxN)
        """
        if len(input_matrix_flat) != self.n * self.n:
            raise ValueError("Input dimensions mismatch")

        # Reconstruct Input Matrix A from flat list
        matrix_a = [input_matrix_flat[i * self.n : (i + 1) * self.n] 
                    for i in range(self.n)]
        
        # Result Matrix C (Accumulators)
        matrix_c = [[0]*self.n for _ in range(self.n)]

        print("[Systolic] Computing MatMul...")
        # Functional Matrix Multiplication
        # In HW, this happens via staggered inputs, but functionally it is:
        for i in range(self.n):          # Row of A
            for j in range(self.n):      # Col of B (Weights)
                acc = 0
                for k in range(self.n):  # Dot Product
                    acc += matrix_a[i][k] * self.weights[k][j]
                matrix_c[i][j] = acc
        
        # Flatten output for the PPU
        output_flat = [val for row in matrix_c for val in row]
        return output_flat

class PPU:
    """
    Post-Processing Unit.
    Handles Activation (ReLU), Quantization, or Scaling.
    """
    def __init__(self, config):
        self.config = config

    def process(self, data, activation="RELU"):
        print(f"[PPU] Processing {len(data)} items with {activation}...")
        processed = []
        for x in data:
            res = x
            if activation == "RELU":
                res = x if x > 0 else 0
            
            # Simple saturation to emulate output bus width (optional)
            # res = min(res, 255) 
            processed.append(res)
        return processed

class Controller:
    """
    The Brain. Fetches instructions and orchestrates blocks.
    Instruction Format: (OPCODE, ADDR_A, ADDR_B, SIZE/Count)
    """
    def __init__(self, memory, array, ppu, config):
        self.memory = memory
        self.array = array
        self.ppu = ppu
        self.config = config

    def execute_program(self, program):
        print("\n--- Starting NPU Execution ---")
        for pc, instr in enumerate(program):
            opcode = instr['op']
            print(f"PC[{pc}]: {opcode}")

            if opcode == 'LOAD_WEIGHTS':
                # Load data from Memory -> Systolic Array Internal Registers
                addr = instr['addr']
                size = self.config.array_size ** 2
                data = self.memory.read_block(addr, size)
                self.array.load_weights(data)

            elif opcode == 'MATMUL':
                # Read Inputs -> Array -> PPU -> Memory
                src_addr = instr['src']
                dst_addr = instr['dst']
                size = self.config.array_size ** 2
                
                # 1. Fetch Input
                inputs = self.memory.read_block(src_addr, size)
                
                # 2. Run Array
                raw_output = self.array.run_matmul(inputs)
                
                # 3. Run PPU
                activated_output = self.ppu.process(raw_output, activation="RELU")
                
                # 4. Write Back
                self.memory.write_block(dst_addr, activated_output)
                print(f"[Controller] Result stored at Address {dst_addr}")
                
            elif opcode == 'HALT':
                print("--- Execution Halted ---")
                break

# ==========================================
# TESTBENCH / USAGE
# ==========================================

def hex_matrix_generator(rows, cols, start_val=1):
    """Helper to create dummy hex strings for testing."""
    vals = []
    for i in range(rows * cols):
        vals.append(f"{(start_val + i) % 255:02X}")
    return " ".join(vals)


import os

# ... [Previous Classes: NPUConfig, Memory, SystolicArray, PPU, Controller] ... 
# ... [Make sure you include the classes from the previous response here] ...

class FileUtils:
    @staticmethod
    def read_hex_file(filename):
        """
        Reads a HEX file (whitespace separated) and returns a string of tokens.
        Compatible with Verilog $readmemh format.
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Could not find {filename}")
        
        with open(filename, 'r') as f:
            content = f.read()
            # Remove simple comments if necessary or just split by whitespace
            tokens = content.split()
        
        # Validate they are hex
        try:
            # Join them back into a single string for the Memory.load_hex_string method
            return " ".join(tokens)
        except Exception as e:
            print(f"Error parsing hex file: {e}")
            return ""

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
# REAL FILE TESTBENCH
# ==========================================

if __name__ == "__main__":
    # 1. Configuration
    config = NPUConfig(array_size=4, data_width=8, mem_depth=1024)
    mem = Memory(config)
    array = SystolicArray(config)
    ppu = PPU(config)
    ctrl = Controller(mem, array, ppu, config)

    # 2. File Paths
    weight_file = "weights.hex"
    input_file = "inputs.hex"
    output_file = "expected_output.hex"

    # 3. Create Dummy Files if they don't exist (for demonstration)
    if not os.path.exists(weight_file):
        print("weights.hex does not exist...")
        # Identity Matrix
        #with open(weight_file, 'w') as f:
        #    f.write("01 00 00 00 00 01 00 00 00 00 01 00 00 00 00 01")
    
    if not os.path.exists(input_file):
        print("inputs.hex does not exist...")
        # 1-16
        #with open(input_file, 'w') as f:
        #    f.write("01 02 03 04 05 06 07 08 09 0A 0B 0C 0D 0E 0F 10")

    # 4. Load Data from Files
    print(f"Loading Weights from {weight_file}...")
    w_data_str = FileUtils.read_hex_file(weight_file)
    mem.load_hex_string(start_addr=0x000, hex_string=w_data_str)

    print(f"Loading Inputs from {input_file}...")
    i_data_str = FileUtils.read_hex_file(input_file)
    mem.load_hex_string(start_addr=0x100, hex_string=i_data_str)

    # 5. Execute Program
    # We assume weights are at 0x000, Inputs at 0x100, Output goes to 0x200
    program = [
        {'op': 'LOAD_WEIGHTS', 'addr': 0x000},
        {'op': 'MATMUL',       'src': 0x100, 'dst': 0x200},
        {'op': 'HALT'}
    ]
    
    ctrl.execute_program(program)

    # 6. Export Results
    # We read 16 bytes (4x4 matrix) from the destination address
    result_hex_str = mem.dump_hex(start_addr=0x200, size=16)
    
    FileUtils.write_hex_file(output_file, result_hex_str)
    
    print("\n--- Test Complete ---")
    print(f"Check {output_file} for the result.")
    print(f"Result Content: {result_hex_str}")
