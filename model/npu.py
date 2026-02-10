import math

class NPUConfig:
    """
    Configuration for the NPU Architecture.
    """
    def __init__(self, array_size=2, data_width=8, mem_depth=1024):
        self.array_size = array_size  # N x N Systolic Array
        self.data_width = data_width  # Bit width (e.g., 8-bit integer)
        self.mem_depth = mem_depth    # Number of addresses in unified memory
        self.max_val = (2**data_width) - 1

class Memory:
    """
    Models the Unified Buffer/SRAM.
    Stores data as integers.
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
            # We assume PPU has already handled quantization to 8-bit.
            # But strictly, memory physically can't hold > 32-bit (or whatever word size).
            self.mem[addr + i] = val

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
        Returns: 32-bit Accumulators (Flat list)
        """
        if len(input_matrix_flat) != self.n * self.n:
            raise ValueError("Input dimensions mismatch")

        # Reconstruct Input Matrix A from flat list
        matrix_a = [input_matrix_flat[i * self.n : (i + 1) * self.n] 
                    for i in range(self.n)]
        
        # Result Matrix C (Accumulators)
        matrix_c = [[0]*self.n for _ in range(self.n)]

        print("[Systolic] Computing MatMul (Accumulating)...")
        for i in range(self.n):          # Row of A
            for j in range(self.n):      # Col of B (Weights)
                acc = 0
                for k in range(self.n):  # Dot Product
                    # Standard Int8 Multiply-Accumulate
                    acc += matrix_a[i][k] * self.weights[k][j]
                matrix_c[i][j] = acc
        
        # Flatten output for the PPU
        output_flat = [val for row in matrix_c for val in row]
        return output_flat

class PPU:
    """
    Post-Processing Unit.
    """
    def __init__(self, config):
        self.config = config

    def process(self, data, scale=1.0, zero_point=0, activation="RELU"):
        """
        Returns a tuple: (activated_data, quantized_data)
        """
        print(f"[PPU] Processing: Scale={scale}, ZP={zero_point}, Act={activation}")
        
        activated_list = []
        quantized_list = []
        
        for x in data:
            # --- Stage 1: Activation ---
            val_act = x
            if activation == "RELU":
                val_act = x if x > 0 else 0
            
            # Save this intermediate state!
            activated_list.append(val_act)
            
            # --- Stage 2: Quantization (Scaling) ---
            val_q = val_act
            if scale != 0:
                val_q = val_q / scale
            
            val_q = val_q + zero_point
            val_q = round(val_q)
            
            # Saturation
            val_q = max(0, min(val_q, self.config.max_val))
            
            quantized_list.append(int(val_q))
            
        return activated_list, quantized_list

class Controller:
    """
    The Brain. Now dumps debug files for every MATMUL operation.
    """
    def __init__(self, memory, array, ppu, config):
        self.memory = memory
        self.array = array
        self.ppu = ppu
        self.config = config

    def _write_debug_hex(self, filename, data):
        """Helper to write a list of integers to a hex file."""
        with open(filename, 'w') as f:
            # Format as 2-digit hex for 8-bit, 8-digit for 32-bit/raw
            hex_str = ' '.join([f'{x:02X}' if x < 256 else f'{x:08X}' for x in data])
            f.write(hex_str + "\n")
        print(f"[Debug] Dumped {filename}")

    def execute_program(self, program):
        print("\n--- Starting NPU Execution ---")
        for pc, instr in enumerate(program):
            opcode = instr['op']
            print(f"PC[{pc}]: {opcode}")

            if opcode == 'LOAD_WEIGHTS':
                addr = instr['addr']
                size = self.config.array_size ** 2
                data = self.memory.read_block(addr, size)
                self.array.load_weights(data)

            elif opcode == 'MATMUL':
                src_addr = instr['src']
                dst_addr = instr['dst']
                scale = instr.get('scale', 1.0)
                zero_point = instr.get('zero_point', 0)
                size = self.config.array_size ** 2
                
                # 1. Fetch Input
                inputs = self.memory.read_block(src_addr, size)
                
                # 2. Run Array (produces RAW 32-bit accumulators)
                raw_output = self.array.run_matmul(inputs)
                
                # --- DUMP 1: Raw Output (Before Activation) ---
                self._write_debug_hex("./hex/result_multiplication.hex", raw_output)
                
                # 3. Run PPU (returns both Activated and Final)
                activated_output, final_output = self.ppu.process(
                    raw_output, scale=scale, zero_point=zero_point
                )
                
                # --- DUMP 2: Activated Output (Before Quantization) ---
                self._write_debug_hex("./hex/result_activated.hex", activated_output)
                
                # --- DUMP 3: Final Output (Quantized) ---
                self._write_debug_hex("./hex/result_final.hex", final_output)
                
                # 4. Write Back Final Result to Memory
                self.memory.write_block(dst_addr, final_output)
                
            elif opcode == 'HALT':
                break


