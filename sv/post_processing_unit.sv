// ============================================================================
// PROJECT: NPU V0.1 (Sovereign AI)
// DESCRIPTION: Scalable Systolic Array Template for INT8 Inference
// ============================================================================

// --- 5. Post-Processing Unit (PPU) ---
module post_processing_unit #(parameter WIDTH = 8)(
    input  logic              clk,
    input  logic [23:0]       data_in,   // High precision input
    input  logic              relu_en,
    output logic [WIDTH-1:0]  data_out   // Quantized 8-bit output
);
    // TODO: Implement ReLU activation and Clipping/Quantization
endmodule
