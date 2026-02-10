// ============================================================================
// PROJECT: NPU V0.1 (Sovereign AI)
// DESCRIPTION: Scalable Systolic Array Template for INT8 Inference
// ============================================================================

// --- 1. Interface Definitions ---
interface npu_axi_stream_if #(parameter WIDTH = 8);
    logic               tvalid;
    logic               tready;
    logic [WIDTH-1:0]   tdata;
    logic               tlast;

    modport slave  (input tvalid, tdata, tlast, output tready);
    modport master (output tvalid, tdata, tlast, input tready);
endinterface
