// ============================================================================
// PROJECT: Tunisian NPU V0.1 (Sovereign AI)
// DESCRIPTION: Scalable Systolic Array Template for INT8 Inference
// ============================================================================

// --- 2. Processing Element (PE) ---
module processing_element #(
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH  = 24  // Extra width to prevent overflow during MAC
)(
    input  logic                     clk,
    input  logic                     rst_n,
    input  logic                     en,        // Enable computation
    
    // Systolic Data Flow
    input  logic [DATA_WIDTH-1:0]    in_west,   // Activation/Input
    input  logic [DATA_WIDTH-1:0]    in_north,  // Weights (or Partial Sums)
    
    output logic [DATA_WIDTH-1:0]    out_east,  // Pass Activation right
    output logic [DATA_WIDTH-1:0]    out_south, // Pass Weight/Result down
    
    // Control and Monitoring
    input  logic                     clear_acc, // Reset accumulator only
    output logic [ACC_WIDTH-1:0]     acc_out    // Final result output
);
    // TODO: Implement Multiply-Accumulate (MAC) logic here
endmodule
