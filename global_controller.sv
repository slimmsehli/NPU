// ============================================================================
// PROJECT: Tunisian NPU V0.1 (Sovereign AI)
// DESCRIPTION: Scalable Systolic Array Template for INT8 Inference
// ============================================================================

// --- 4. Global Controller ---
module global_controller (
    input  logic clk,
    input  logic rst_n,
    
    // Command Interface (Simplified)
    input  logic [3:0]  opcode,
    input  logic        start,
    output logic        busy,
    output logic        done,

    // Internal Control Signals
    output logic        sram_read_en,
    output logic        pe_array_en,
    output logic        ppu_en
);
    // TODO: Implement FSM to manage LOAD -> COMPUTE -> STORE cycles
endmodule
