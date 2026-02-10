// ============================================================================
// PROJECT: NPU V0.1 (Sovereign AI)
// DESCRIPTION: Scalable Systolic Array Template for INT8 Inference
// ============================================================================

// --- 3. Systolic Array (The Grid) ---
module systolic_array #(
    parameter ARRAY_SIZE = 4,
    parameter DATA_WIDTH = 8
)(
    input  logic clk,
    input  logic rst_n,
    
    // Input buses from SRAM/Buffers
    input  logic [DATA_WIDTH-1:0] row_in [ARRAY_SIZE-1:0],
    input  logic [DATA_WIDTH-1:0] col_in [ARRAY_SIZE-1:0],
    
    // Output results to PPU
    output logic [(DATA_WIDTH*3)-1:0] array_out [ARRAY_SIZE-1:0]
);
    // TODO: Generate block to instantiate ARRAY_SIZE * ARRAY_SIZE PEs
    // and wire them together in a grid.
endmodule
