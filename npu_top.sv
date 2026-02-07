// ============================================================================
// PROJECT: Tunisian NPU V0.1 (Sovereign AI)
// DESCRIPTION: Scalable Systolic Array Template for INT8 Inference
// ============================================================================
/* verilator lint_off DECLFILENAME */
// --- 6. TOP LEVEL NPU WRAPPER ---
module npu_top #(
    parameter ARRAY_SIZE = 4,
    parameter DATA_WIDTH = 8
)(
    input logic clk,
    input logic rst_n,

    // AXI4-Stream Slave (for Weights/Inputs)
    npu_axi_stream_if.slave  s_axis,
    
    // AXI4-Stream Master (for Results)
    npu_axi_stream_if.master m_axis
);

    // Internal Signal Declarations
    logic [DATA_WIDTH-1:0] row_bus [ARRAY_SIZE-1:0];
    logic [DATA_WIDTH-1:0] col_bus [ARRAY_SIZE-1:0];

    // Block 1: Global Controller
    global_controller ctrl_inst (.*);

    // Block 2: Systolic Array
    systolic_array #(ARRAY_SIZE, DATA_WIDTH) array_inst (
        .clk(clk),
        .rst_n(rst_n),
        .row_in(row_bus),
        .col_in(col_bus),
        .array_out() // Connect to PPU
    );

    // Block 3: PPU (One per column usually)
    generate
        for (genvar i=0; i<ARRAY_SIZE; i++) begin : ppu_gen
            post_processing_unit ppu_inst (.*);
        end
    endgenerate
    
    initial begin
		$dumpfile("waves.vcd");
        $dumpvars(0, npu_top);
    	#10;
    	$finish;
    end

endmodule
