
// Testbench for Sthe systolic Array 


module top();

    parameter DATA_WIDTH = 8;
    parameter ACC_WIDTH = 32;
    parameter FRAC_BITS = 8;
    parameter CLK_PERIOD = 10;
    
    reg clk;
    reg rst;
    reg start;
    
    // Input matrices
    reg [DATA_WIDTH-1:0] matrix_a [0:8];
    reg [DATA_WIDTH-1:0] matrix_b [0:8];
    
    // VPU control
    reg vpu_enable;
    reg [2:0] vpu_operation;
    reg [ACC_WIDTH-1:0] bias_value;
    reg [ACC_WIDTH-1:0] scale_factor;
    
    // Outputs
    wire [ACC_WIDTH-1:0] raw_result_00, raw_result_01, raw_result_02;
    wire [ACC_WIDTH-1:0] raw_result_10, raw_result_11, raw_result_12;
    wire [ACC_WIDTH-1:0] raw_result_20, raw_result_21, raw_result_22;
    
    wire [ACC_WIDTH-1:0] result_00, result_01, result_02;
    wire [ACC_WIDTH-1:0] result_10, result_11, result_12;
    wire [ACC_WIDTH-1:0] result_20, result_21, result_22;
    
    wire controller_done;
    wire vpu_valid;
    
    // VPU operation codes
    localparam OP_PASSTHROUGH = 3'd0;
    localparam OP_RELU        = 3'd1;
    localparam OP_BIAS_ADD    = 3'd2;
    localparam OP_SCALE       = 3'd3;
    localparam OP_MAX_POOL    = 3'd4;
    localparam OP_AVG_POOL    = 3'd5;
    localparam OP_SIGMOID     = 3'd6;
    localparam OP_TANH        = 3'd7;
    
    // Clock generation
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end
    
    // memory load
    initial begin
      $readmemh("../memories/weights_L0.hex", matrix_a);
      $readmemh("../memories/inputs.hex", matrix_b);
  	end
    
    // Instantiate DUT
    systolic_with_vpu #(
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .FRAC_BITS(FRAC_BITS)
    ) dut (
        .clk(clk),
        .rst(rst),
        .start(start),
        .matrix_a(matrix_a),
        .matrix_b(matrix_b),
        .vpu_enable(vpu_enable),
        .vpu_operation(vpu_operation),
        .bias_value(bias_value),
        .scale_factor(scale_factor),
        .raw_result_00(raw_result_00),
        .raw_result_01(raw_result_01),
        .raw_result_02(raw_result_02),
        .raw_result_10(raw_result_10),
        .raw_result_11(raw_result_11),
        .raw_result_12(raw_result_12),
        .raw_result_20(raw_result_20),
        .raw_result_21(raw_result_21),
        .raw_result_22(raw_result_22),
        .result_00(result_00),
        .result_01(result_01),
        .result_02(result_02),
        .result_10(result_10),
        .result_11(result_11),
        .result_12(result_12),
        .result_20(result_20),
        .result_21(result_21),
        .result_22(result_22),
        .controller_done(controller_done),
        .vpu_valid(vpu_valid)
    );
    
    // Test procedure
    initial begin
        $display("=======================================================");
        $display("Systolic Array with Vector Processing Unit - Test Suite");
        $display("=======================================================\n");
        
        // Initialize
        rst = 1;
        start = 0;
        vpu_enable = 0;
        vpu_operation = OP_PASSTHROUGH;
        bias_value = 0;
        scale_factor = 0;
        
        // Initialize matrices
        // Matrix A
        //matrix_a[0] = 1; matrix_a[1] = 2; matrix_a[2] = 3;
        //matrix_a[3] = 4; matrix_a[4] = 5; matrix_a[5] = 6;
        //matrix_a[6] = 7; matrix_a[7] = 8; matrix_a[8] = 9;
        
        // Matrix B
        //matrix_b[0] = 9; matrix_b[1] = 8; matrix_b[2] = 7;
        //matrix_b[3] = 6; matrix_b[4] = 5; matrix_b[5] = 4;
        //matrix_b[6] = 3; matrix_b[7] = 2; matrix_b[8] = 1;
        
        $display("Matrix A:");
        $display("[%0d %0d %0d]", matrix_a[0], matrix_a[1], matrix_a[2]);
        $display("[%0d %0d %0d]", matrix_a[3], matrix_a[4], matrix_a[5]);
        $display("[%0d %0d %0d]", matrix_a[6], matrix_a[7], matrix_a[8]);
        
        $display("\nMatrix B:");
        $display("[%0d %0d %0d]", matrix_b[0], matrix_b[1], matrix_b[2]);
        $display("[%0d %0d %0d]", matrix_b[3], matrix_b[4], matrix_b[5]);
        $display("[%0d %0d %0d]", matrix_b[6], matrix_b[7], matrix_b[8]);
        
        // Reset
        #(CLK_PERIOD*2);
        rst = 0;
        #(CLK_PERIOD);
        
        // Start systolic array computation
        start = 1;
        
        // Wait for completion
        wait(controller_done);
        #(CLK_PERIOD*5);
        
        // Display raw results
        $display("\n--- Raw Systolic Array Output (C = A × B) ---");
        $display("[%0d %0d %0d]", raw_result_00, raw_result_01, raw_result_02);
        $display("[%0d %0d %0d]", raw_result_10, raw_result_11, raw_result_12);
        $display("[%0d %0d %0d]", raw_result_20, raw_result_21, raw_result_22);
        $display("Expected: [30 24 18] [84 69 54] [138 114 90]");
        
        // TEST 1: Pass-through
        $display("\n\n=== TEST 1: Pass-through ===");
        vpu_operation = OP_PASSTHROUGH;
        vpu_enable = 1;
        #(CLK_PERIOD);
        vpu_enable = 0;
        #(CLK_PERIOD*2);
        $display("Output (should be same as raw):");
        $display("[%0d %0d %0d]", result_00, result_01, result_02);
        $display("[%0d %0d %0d]", result_10, result_11, result_12);
        $display("[%0d %0d %0d]", result_20, result_21, result_22);
        
        // TEST 2: ReLU activation
        $display("\n\n=== TEST 2: ReLU Activation ===");
        vpu_operation = OP_RELU;
        vpu_enable = 1;
        #(CLK_PERIOD);
        vpu_enable = 0;
        #(CLK_PERIOD*2);
        $display("Output (max(0, x)):");
        $display("[%0d %0d %0d]", result_00, result_01, result_02);
        $display("[%0d %0d %0d]", result_10, result_11, result_12);
        $display("[%0d %0d %0d]", result_20, result_21, result_22);
        
        // TEST 3: Bias addition
        $display("\n\n=== TEST 3: Bias Addition (bias = 10) ===");
        vpu_operation = OP_BIAS_ADD;
        bias_value = 10;
        vpu_enable = 1;
        #(CLK_PERIOD);
        vpu_enable = 0;
        #(CLK_PERIOD*2);
        $display("Output (raw + 10):");
        $display("[%0d %0d %0d]", result_00, result_01, result_02);
        $display("[%0d %0d %0d]", result_10, result_11, result_12);
        $display("[%0d %0d %0d]", result_20, result_21, result_22);
        $display("Expected: [40 34 28] [94 79 64] [148 124 100]");
        
        // TEST 4: Scaling
        $display("\n\n=== TEST 4: Scaling (scale = 0.5, factor = 128) ===");
        vpu_operation = OP_SCALE;
        scale_factor = 128; // 0.5 in fixed point (128/256 = 0.5)
        vpu_enable = 1;
        #(CLK_PERIOD);
        vpu_enable = 0;
        #(CLK_PERIOD*2);
        $display("Output (raw × 0.5):");
        $display("[%0d %0d %0d]", result_00, result_01, result_02);
        $display("[%0d %0d %0d]", result_10, result_11, result_12);
        $display("[%0d %0d %0d]", result_20, result_21, result_22);
        $display("Expected (approx): [15 12 9] [42 34 27] [69 57 45]");
        
        // TEST 5: Max Pooling
        $display("\n\n=== TEST 5: Max Pooling (2x2) ===");
        vpu_operation = OP_MAX_POOL;
        vpu_enable = 1;
        #(CLK_PERIOD);
        vpu_enable = 0;
        #(CLK_PERIOD*2);
        $display("Output (2x2 max pool, 4 valid outputs):");
        $display("[%0d %0d %0d]", result_00, result_01, result_02);
        $display("[%0d %0d %0d]", result_10, result_11, result_12);
        $display("[%0d %0d %0d]", result_20, result_21, result_22);
        $display("Expected: [84 69 0] [138 114 0] [0 0 0]");
        
        // TEST 6: Average Pooling
        $display("\n\n=== TEST 6: Average Pooling (2x2) ===");
        vpu_operation = OP_AVG_POOL;
        vpu_enable = 1;
        #(CLK_PERIOD);
        vpu_enable = 0;
        #(CLK_PERIOD*2);
        $display("Output (2x2 avg pool, 4 valid outputs):");
        $display("[%0d %0d %0d]", result_00, result_01, result_02);
        $display("[%0d %0d %0d]", result_10, result_11, result_12);
        $display("[%0d %0d %0d]", result_20, result_21, result_22);
        $display("Expected (approx): [51 42 0] [103 85 0] [0 0 0]");
        
        // TEST 7: Sigmoid approximation
        $display("\n\n=== TEST 7: Sigmoid Approximation ===");
        vpu_operation = OP_SIGMOID;
        vpu_enable = 1;
        #(CLK_PERIOD);
        vpu_enable = 0;
        #(CLK_PERIOD*2);
        $display("Output (sigmoid approx):");
        $display("[%0d %0d %0d]", result_00, result_01, result_02);
        $display("[%0d %0d %0d]", result_10, result_11, result_12);
        $display("[%0d %0d %0d]", result_20, result_21, result_22);
        
        // TEST 8: Tanh approximation
        $display("\n\n=== TEST 8: Tanh Approximation ===");
        vpu_operation = OP_TANH;
        vpu_enable = 1;
        #(CLK_PERIOD);
        vpu_enable = 0;
        #(CLK_PERIOD*2);
        $display("Output (tanh approx):");
        $display("[%0d %0d %0d]", result_00, result_01, result_02);
        $display("[%0d %0d %0d]", result_10, result_11, result_12);
        $display("[%0d %0d %0d]", result_20, result_21, result_22);
        
        $display("\n=======================================================");
        $display("All tests completed!");
        $display("=======================================================");
        $finish;
    end

endmodule
