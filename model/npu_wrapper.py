import math
# Import your existing NPU classes (assuming they are in npu_design.py)
from npu import NPUConfig, Memory, SystolicArray, PPU, Controller

# ... (Import classes as before) ...

class NPUDriver:
    # ... (Init method remains the same) ...
    def __init__(self, array_size=8, data_width=8, mem_depth=2048):
        # ... same initialization code ...
        self.config = NPUConfig(array_size, data_width, mem_depth)
        self.mem = Memory(self.config)
        self.array = SystolicArray(self.config)
        self.ppu = PPU(self.config)
        self.ctrl = Controller(self.mem, self.array, self.ppu, self.config)
        
        # Memory Map
        self.ADDR_WEIGHTS = 0x000
        self.ADDR_BUF_A   = 0x100
        self.ADDR_BUF_B   = 0x200
        self.BLOCK_SIZE   = self.config.array_size ** 2

    def _load_data_to_ram(self, addr, data_list):
        # ... same helper ...
        hex_str = " ".join([f"{x:02X}" for x in data_list])
        self.mem.load_hex_string(addr, hex_str)

    def run_inference(self, input_data, layers):
        """
        Runs a full multi-layer inference.
        """
        print(f"\n=== Starting Deep Neural Network Inference ({len(layers)} Layers) ===")
        
        # 1. Load Initial Input into Buffer A
        print(f"[Driver] Loading Network Input to Buffer A (0x{self.ADDR_BUF_A:X})")
        self._load_data_to_ram(self.ADDR_BUF_A, input_data)
        
        # Pointers for Ping-Pong Buffering
        current_src = self.ADDR_BUF_A
        current_dst = self.ADDR_BUF_B
        
        # 2. Loop through layers
        # 'i' is your LAYER NUMBER (0, 1, 2...)
        for i, layer in enumerate(layers):
            layer_num = i + 1  # Humans usually count from 1
            print(f"\n--- [Driver] Processing Layer #{layer_num} ---")
            
            # A. Extract Layer Params
            weights = layer['weights']
            scale = layer.get('scale', 1.0)
            zero_point = layer.get('zero_point', 0)
            
            # B. Load Weights
            self._load_data_to_ram(self.ADDR_WEIGHTS, weights)
            
            # C. Generate Micro-Program
            # We can now inject the 'layer_id' into the instruction dict if we want
            # the Controller to see it (e.g. for naming debug files).
            program = [
                {'op': 'LOAD_WEIGHTS', 'addr': self.ADDR_WEIGHTS},
                {
                    'op': 'MATMUL', 
                    'src': current_src, 
                    'dst': current_dst, 
                    'scale': scale, 
                    'zero_point': zero_point,
                    'layer_id': layer_num  # <--- PASSING PARAMETER HERE
                }
            ]
            
            # D. Execute
            self.ctrl.execute_program(program, layer_num)
            
            # E. Swap Buffers
            current_src, current_dst = current_dst, current_src

        # 3. Retrieve Final Result
        # Note: current_src is where the last result was written
        print(f"\n[Driver] Inference Done. Reading result from 0x{current_src:X}")
        return self.mem.read_block(current_src, self.BLOCK_SIZE)

# ==========================================
# USAGE EXAMPLE (The "User" Code)
# ==========================================
"""
if __name__ == "__main__":
    driver = NPUDriver(array_size=8, data_width=8, mem_depth=2048)
    
    # --- Define Data (4x4 flattened) ---
    # Input: Identity-like with some values
    user_input = [
        10, 0, 0, 0,
        0, 10, 0, 0,
        0, 0, 10, 0,
        0, 0, 0, 10
    ]
    
    # --- Define Model (2 Layers) ---
    # Layer 1: Doubles the value (Weight=2, Scale=1)
    weights_L1 = [
        2, 0, 0, 0,
        0, 2, 0, 0,
        0, 0, 2, 0,
        0, 0, 0, 2
    ]
    
    # Layer 2: Halves the value (Weight=1, Scale=2.0)
    weights_L2 = [
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    ]
    
    my_model = [
        {'weights': weights_L1, 'scale': 1.0, 'zero_point': 0}, # 10 * 2 = 20
        {'weights': weights_L2, 'scale': 2.0, 'zero_point': 0}  # 20 * 1 / 2 = 10
    ]
    
    # --- Run ---
    final_output = driver.run_inference(user_input, my_model)
    
    print("\nFinal Network Output:", final_output)
"""


