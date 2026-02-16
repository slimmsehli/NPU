
// Testbench for Sthe systolic Array 


module npu_top
	#(
		parameter DATA_WIDTH = 8,
		parameter ACC_WIDTH = 32,
		parameter FRAC_BITS = 8,
		parameter CLK_PERIOD = 10,
		parameter LENGTH = 9,
		parameter LAYERS = 3
	)
	(
    // general input
    input clk,
    input rst,
    input start_p,
    // input matrices
    input reg [DATA_WIDTH-1:0] mat_inputs [0:LENGTH-1],
    input reg [DATA_WIDTH-1:0] mat_weights [0:LAYERS-1][0:LENGTH-1],
    
    // output
    output [ACC_WIDTH-1:0] raw_result [0:LAYERS-1][0:LENGTH-1],
    output [ACC_WIDTH-1:0] result [0:LAYERS-1][0:LENGTH-1],
    output [LAYERS-1:0] controller_done,
    output [LAYERS-1:0] vpu_valid
);    
    
		// temp matrix for input of all layers
    reg [DATA_WIDTH-1:0] matrix_a [0:LENGTH-1];
    reg [DATA_WIDTH-1:0] matrix_b [0:LENGTH-1];
   
    // VPU control signals
    reg [LAYERS-1:0] vpu_enable ;
    reg [2:0] vpu_operation [0:LAYERS-1];
    reg [ACC_WIDTH-1:0] bias_value [0:LAYERS-1];
    reg [ACC_WIDTH-1:0] scale_factor [0:LAYERS-1];
  	
  	// internal flags for the layers FSM
    reg [2:0] current_layer;
    reg [LAYERS-1:0] start_layer;
    reg [3:0] status;
    
    // VPU codes
    localparam OP_PASSTHROUGH = 3'd0;
    localparam OP_RELU        = 3'd1;
    localparam OP_BIAS_ADD    = 3'd2;
    localparam OP_SCALE       = 3'd3;
    localparam OP_MAX_POOL    = 3'd4;
    localparam OP_AVG_POOL    = 3'd5;
    localparam OP_SIGMOID     = 3'd6;
    localparam OP_TANH        = 3'd7;

    // ######################################
    // npu layers
    genvar i;
    generate
    	for (i=0;i<LAYERS;i++) begin : npu_layers
    		npu_layer #(
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .FRAC_BITS(FRAC_BITS),
        .LENGTH(LENGTH)
				) layer_i (
				    .clk(clk),
				    .rst(rst),
				    .start(start_layer[i]),
				    .matrix_a(matrix_a),
				    .matrix_b(matrix_b),
				    .vpu_enable(vpu_enable[i]),
				    .vpu_operation(vpu_operation[i]),
				    .bias_value(bias_value[i]),
				    .scale_factor(scale_factor[i]),
				    .raw_result(raw_result[i]),
				    .result(result[i]),
				    .controller_done(controller_done[i]),
				    .vpu_valid(vpu_valid[i])
				);
    	end
    endgenerate
    
    always @(posedge clk, posedge rst) begin
    	if (rst==1) begin
    		// Initialize
        start_layer = 3'b000;
        vpu_enable = 3'b000;
        bias_value = {0, 0, 0};
        scale_factor = {0, 0, 0};
        vpu_operation = {OP_RELU, OP_RELU, OP_RELU};
        current_layer = 0;
        status = 0;
    	end
    	else begin
    		case (status)
    			0: begin //IDLE 
    				if (start_p == 1) status = 1;
    			end 
    			1: begin // starting
    				for (integer i=0; i<LAYERS; i++) begin : layer_loop
				    	current_layer = i;
				    	if (i==0)
				    		matrix_a = mat_inputs;
				    	else
								matrix_a = raw_result[current_layer-1];
								
								matrix_b = mat_weights[current_layer];
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
								vpu_enable[current_layer] = 1;
								start_layer[current_layer] = 1;
								#(CLK_PERIOD);
								 vpu_enable[current_layer] = 0;
								
								
								// Wait for completion
								wait(controller_done[current_layer]);
								#(CLK_PERIOD*5);
								start_layer[current_layer] = 0;
								
								// Display raw results
								$display("\n--- Layer 3 Raw Systolic Array Output (C = A × B) ---");
								for (integer i=1; i<LENGTH+1;i++) begin
									$write(" %0d ", raw_result[current_layer][i-1]);
									if (i%3==0) $write("\n");
								end
				    end
				    status = 0;
    			end
    			2: begin end
    			3: begin end
    			default: begin end
    		endcase
    	end
    end
endmodule



module top;
		parameter DATA_WIDTH = 8;
		parameter ACC_WIDTH = 32;
		parameter FRAC_BITS = 8;
		parameter CLK_PERIOD = 10;
		parameter LENGTH = 9;
		parameter LAYERS = 3;
		
		reg clk;
    reg rst;
    reg start_p;
   
    // input matrices
    reg [DATA_WIDTH-1:0] mat_inputs [0:LENGTH-1];
    reg [DATA_WIDTH-1:0] mat_weights [0:LAYERS-1][0:LENGTH-1];
    
    // output
    wire [ACC_WIDTH-1:0] raw_result [0:LAYERS-1][0:LENGTH-1];
    wire [ACC_WIDTH-1:0] result [0:LAYERS-1][0:LENGTH-1];
    wire [LAYERS-1:0] controller_done;
    wire [LAYERS-1:0] vpu_valid;


	// ######################################
	// Clock
	initial begin
		  clk = 0;
		  forever #(CLK_PERIOD/2) clk = ~clk;
	end
	
	// memory load
    initial begin
      $readmemh("../memories/inputs.hex", mat_inputs);
      $readmemh("../memories/weights_L0.hex", mat_weights[0]);
      $readmemh("../memories/weights_L1.hex", mat_weights[1]);
      $readmemh("../memories/weights_L2.hex", mat_weights[2]);
  	end
	
	// ######################################
    // Test procedure
		initial begin
			$display("=======================================================");
			$display("Systolic Array with Vector Processing Unit - Test Suite");
			$display("=======================================================\n");
			start_p = 0;
			rst = 1'b1;
			start_p = 0;
			// Reset
			#(CLK_PERIOD*2);
			rst = 0;
			#(CLK_PERIOD*5);
			start_p = 1;
			repeat (50) @(posedge clk);
			$finish;
		end
	
	// ######################################
    // Final loop
    integer fd;
    final begin
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
        
    end
    
    // ######################################
    npu_top
			#(
				.DATA_WIDTH(DATA_WIDTH), //parameter DATA_WIDTH = 8,
				.ACC_WIDTH(ACC_WIDTH), //parameter ACC_WIDTH = 32,
				.FRAC_BITS(FRAC_BITS), //parameter FRAC_BITS = 8,
				.CLK_PERIOD(CLK_PERIOD), //parameter CLK_PERIOD = 10,
				.LENGTH(LENGTH), //parameter LENGTH = 9,
				.LAYERS(LAYERS) //parameter LAYERS = 3
			)
			npu_inst
			(
				// general input
				.clk(clk), //input clk,
				.rst(rst), //input rst,
				.start_p(start_p), //input start_p,
				// input matrices
				.mat_inputs(mat_inputs), //input reg [DATA_WIDTH-1:0] mat_inputs [0:LENGTH-1],
				.mat_weights(mat_weights), //input reg [DATA_WIDTH-1:0] mat_weights [0:LAYERS-1][0:LENGTH-1],
				
				// output
				.raw_result(raw_result), //output [ACC_WIDTH-1:0] raw_result [0:LAYERS-1][0:LENGTH-1],
				.result(result), //output [ACC_WIDTH-1:0] result [0:LAYERS-1][0:LENGTH-1],
				.controller_done(controller_done), //output [LAYERS-1:0] controller_done,
				.vpu_valid(vpu_valid) //output [LAYERS-1:0] vpu_valid
		);  

endmodule
