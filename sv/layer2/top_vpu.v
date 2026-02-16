
// Testbench for Sthe systolic Array 


module top();

    parameter DATA_WIDTH = 8;
    parameter ACC_WIDTH = 32;
    parameter FRAC_BITS = 8;
    parameter CLK_PERIOD = 10;
    parameter LENGTH = 9;
    parameter LAYERS = 3;
    
    reg clk;
    reg rst;
    reg [LAYERS-1:0] start;
    
    // Input matrices
    reg [DATA_WIDTH-1:0] matrix_a [0:8];
    reg [DATA_WIDTH-1:0] matrix_b [0:8];
    
    reg [DATA_WIDTH-1:0] mat_inputs [0:8];
    reg [DATA_WIDTH-1:0] mat_weights_0 [0:8];
    reg [DATA_WIDTH-1:0] mat_weights_1 [0:8];
    reg [DATA_WIDTH-1:0] mat_weights_2 [0:8];
    
    // VPU control
    reg [LAYERS-1:0] vpu_enable ;
    reg [2:0] vpu_operation [0:LAYERS-1];
    reg [ACC_WIDTH-1:0] bias_value [0:LAYERS-1];
    reg [ACC_WIDTH-1:0] scale_factor [0:LAYERS-1];
    
    // Outputs
    wire [ACC_WIDTH-1:0] raw_result [0:LAYERS-1][0:LENGTH-1];
  	
  	wire [ACC_WIDTH-1:0] result [0:LAYERS-1][0:LENGTH-1];
    
    wire [LAYERS-1:0] controller_done;
    wire [LAYERS-1:0] vpu_valid;
    
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
      $readmemh("../memories/inputs.hex", mat_inputs);
      $readmemh("../memories/weights_L0.hex", mat_weights_0);
      $readmemh("../memories/weights_L1.hex", mat_weights_1);
      $readmemh("../memories/weights_L2.hex", mat_weights_2);
  	end
    
    // Instantiate DUT
    npu_layer #(
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .FRAC_BITS(FRAC_BITS),
        .LENGTH(LENGTH)
    ) dut1 (
        .clk(clk),
        .rst(rst),
        .start(start[0]),
        .matrix_a(matrix_a),
        .matrix_b(matrix_b),
        .vpu_enable(vpu_enable[0]),
        .vpu_operation(vpu_operation[0]),
        .bias_value(bias_value[0]),
        .scale_factor(scale_factor[0]),
        .raw_result(raw_result[0]),
        .result(result[0]),
        .controller_done(controller_done[0]),
        .vpu_valid(vpu_valid[0])
    );
    
    // Instantiate DUT
    npu_layer #(
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .FRAC_BITS(FRAC_BITS),
        .LENGTH(LENGTH)
    ) dut2 (
        .clk(clk),
        .rst(rst),
        .start(start[1]),
        .matrix_a(matrix_a),
        .matrix_b(matrix_b),
        .vpu_enable(vpu_enable[1]),
        .vpu_operation(vpu_operation[1]),
        .bias_value(bias_value[1]),
        .scale_factor(scale_factor[1]),
        .raw_result(raw_result[1]),
        .result(result[1]),
        .controller_done(controller_done[1]),
        .vpu_valid(vpu_valid[1])
    );
    
    // Instantiate DUT
    npu_layer #(
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .FRAC_BITS(FRAC_BITS),
        .LENGTH(LENGTH)
    ) dut3 (
        .clk(clk),
        .rst(rst),
        .start(start[2]),
        .matrix_a(matrix_a),
        .matrix_b(matrix_b),
        .vpu_enable(vpu_enable[2]),
        .vpu_operation(vpu_operation[2]),
        .bias_value(bias_value[2]),
        .scale_factor(scale_factor[2]),
        .raw_result(raw_result[2]),
        .result(result[2]),
        .controller_done(controller_done[2]),
        .vpu_valid(vpu_valid[2])
    );
    
    integer fd;
    // Test procedure
    initial begin
        $display("=======================================================");
        $display("Systolic Array with Vector Processing Unit - Test Suite");
        $display("=======================================================\n");
        
        // Initialize
        rst = 1;
        start = 3'b000;
        vpu_enable = 3'b000;
        bias_value = {0, 0, 0};
        scale_factor = {0, 0, 0};
        vpu_operation = {OP_RELU, OP_RELU, OP_RELU};
        
        // Reset
        #(CLK_PERIOD*2);
        rst = 0;
        #(CLK_PERIOD);
        
        
        //###########################################
        // first layer
        matrix_a = mat_inputs;
        matrix_b = mat_weights_0;
        $display("\n--- Layer 1 ");
        $display("Matrix A:");
        for (integer i=1; i<LENGTH+1;i++) begin
        	$write(" %0d ", matrix_a[i-1]);
        	if (i%3==0) $write("\n");
        end
        
        $display("\nMatrix B:");
        for (integer i=1; i<LENGTH+1;i++) begin
        	$write(" %0d ", matrix_b[i-1]);
        	if (i%3==0) $write("\n");
        end
        
        // Start systolic array computation
        vpu_enable = 3'b001;
        start = 3'b001;
        #(CLK_PERIOD);
        vpu_enable = 3'b000;
        
        
        // Wait for completion
        wait(controller_done[0]);
        #(CLK_PERIOD*5);
        start = 3'b000;
        
        // Display raw results
        $display("\n--- Layer 1 Raw Systolic Array Output (C = A × B) ---");
        for (integer i=1; i<LENGTH+1;i++) begin
        	$write(" %0d ", raw_result[0][i-1]);
        	if (i%3==0) $write("\n");
        end
        
        //###########################################
        // second layer
        matrix_a = raw_result[0];
        matrix_b = mat_weights_1;
        $display("\n--- Layer 2 ");
        $display("Matrix A:");
        for (integer i=1; i<LENGTH+1;i++) begin
        	$write(" %0d ", matrix_a[i-1]);
        	if (i%3==0) $write("\n");
        end
        
        $display("\nMatrix B:");
        for (integer i=1; i<LENGTH+1;i++) begin
        	$write(" %0d ", matrix_b[i-1]);
        	if (i%3==0) $write("\n");
        end
        
        // Start systolic array computation
        vpu_enable = 3'b010;
        start = 3'b010;
        #(CLK_PERIOD);
        vpu_enable = 3'b000;
        
        
        // Wait for completion
        wait(controller_done[1]);
        #(CLK_PERIOD*5);
        start = 3'b000;
        
        // Display raw results
        $display("\n--- Layer 2 Raw Systolic Array Output (C = A × B) ---");
        for (integer i=1; i<LENGTH+1;i++) begin
        	$write(" %0d ", raw_result[1][i-1]);
        	if (i%3==0) $write("\n");
        end
        
        //###########################################
        // Third layer
        matrix_a = raw_result[1];
        matrix_b = mat_weights_2;
        $display("\n--- Layer 3 ");
        $display("Matrix A:");
        for (integer i=1; i<LENGTH+1;i++) begin
        	$write(" %0d ", matrix_a[i-1]);
        	if (i%3==0) $write("\n");
        end
        
        $display("\nMatrix B:");
        for (integer i=1; i<LENGTH+1;i++) begin
        	$write(" %0d ", matrix_b[i-1]);
        	if (i%3==0) $write("\n");
        end
        
        // Start systolic array computation
        vpu_enable = 3'b100;
        start = 3'b100;
        #(CLK_PERIOD);
        vpu_enable = 3'b000;
        
        
        // Wait for completion
        wait(controller_done[2]);
        #(CLK_PERIOD*5);
        start = 3'b000;
        
        // Display raw results
        $display("\n--- Layer 3 Raw Systolic Array Output (C = A × B) ---");
        for (integer i=1; i<LENGTH+1;i++) begin
        	$write(" %0d ", raw_result[2][i-1]);
        	if (i%3==0) $write("\n");
        end
        
        
        
        
        $display("\n=======================================================");
        $display("All tests completed!");
        $display("=======================================================");
        // Display raw results
        $display("\n\n--- Input Matrix ---\n");
        for (integer i=1; i<LENGTH+1;i++) begin
        	$write(" %0d ", mat_inputs[i-1]);
        	if (i%3==0) $write("\n");
        end
        // Display raw results
        $display("\n\n--- Output Matrix ---\n");
        for (integer i=1; i<LENGTH+1;i++) begin
        	$write(" %0d ", raw_result[2][i-1]);
        	if (i%3==0) $write("\n");
        end
        
        fd = $fopen("../memories/npu_output.hex", "w");
				for (integer i = 0; i < LENGTH; i++) begin
					$fdisplay(fd, "%01h", raw_result[2][i]);
				end
				$fclose(fd);        
        $finish;
    end

endmodule
