
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
    output [LAYERS-1:0] vpu_valid,
    output reg done
);    
    
		// temp matrix for input of all layers
    reg [DATA_WIDTH-1:0] matrix_a [0:LENGTH-1];
    reg [DATA_WIDTH-1:0] matrix_b [0:LENGTH-1];
   
    // VPU control signals
    reg [LAYERS-1:0] vpu_enable ;
    reg [LAYERS-1:0] en_bias;
    reg [LAYERS-1:0] en_activation;
    reg [LAYERS-1:0] en_scale;
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
				    .en_bias(en_bias[i]),
    				.en_activation(en_activation[i]),
    				.en_scale(en_scale[i]),
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
        scale_factor = {1, 1, 1};
        vpu_operation = {OP_RELU, OP_RELU, OP_RELU};
        en_bias = 3'b111;
        en_activation = 3'b111;
        en_scale = 3'b111;
        current_layer = 0;
        status = 0;
        done = 1'b0;
    	end
    	else begin
    		case (status)
    			0: begin //IDLE 
    				done = 1'b0;
    				if (start_p == 1) status = 1; // status=1 set current layer
    			end 
    			1: begin // load matrices 
    				if (current_layer==0)
				    		matrix_a = mat_inputs;
			    	else begin //@NOTE fix from directly assiging an array to another 
							//matrix_a = result[current_layer-1];
							for (integer i=0; i<LENGTH; i++) begin
								matrix_a[i] = result[current_layer-1][i];
							end
						end
							matrix_b = mat_weights[current_layer];
						status = 2;
    			end
    			2: begin // enable layer and vpu calculation and move to wait 
    				vpu_enable[current_layer] = 1;
						start_layer[current_layer] = 1;
						status = 3; //wait state
    			end
    			3 : begin
    				
    				if (controller_done[current_layer]) begin
    					status = 4; // move to next state only when the systolic array was completed
    					end
    			end
    			4 : begin // wait state for layer deactivation
    				start_layer[current_layer] = 0;
    				vpu_enable[current_layer] = 0;
    				if (current_layer==LAYERS-1)
    					status = 5; // if we are at the last layer we go out
    				else begin
    					current_layer = current_layer+1; 
    					status = 1; // move to the next layer (load matrices) and update layer counter 
    				end
    			end
    			5 : begin // end of all layers
    				done = 1'b1;
    				status = 0;
    			end
    			10: begin // starting
    				for (integer i=0; i<LAYERS; i++) begin : layer_loop
				    	
				    	// EDGE : set current layer
				    	current_layer = i;
				    	if (i==0)
				    		matrix_a = mat_inputs;
				    	else begin //@NOTE fix from directly assiging an array to another 
				    		//matrix_a = result[current_layer-1];
				    		for (integer i=0; i<LENGTH; i++) begin
								matrix_a[i] = result[current_layer-1][i];
								end
							end
								
								
								matrix_b = mat_weights[current_layer];
								/*$display("\n--- Layer 3 ");
								$display("Matrix A:");
								for (integer i=1; i<LENGTH+1;i++) begin
									$write(" %0d ", matrix_a[i-1]);
									if (i%3==0) $write("\n");
								end*/
								
								/*$display("\nMatrix B:");
								for (integer i=1; i<LENGTH+1;i++) begin
									$write(" %0d ", matrix_b[i-1]);
									if (i%3==0) $write("\n");
								end*/
								
								// Start systolic array computation
								vpu_enable[current_layer] = 1;
								start_layer[current_layer] = 1;
								#(CLK_PERIOD);
								
								// EDGE : deactivate the vpu
								 vpu_enable[current_layer] = 0;
								
								// EDGE : waiting for systollic array to complete
								// Wait for completion
								wait(controller_done[current_layer]);
								#(CLK_PERIOD*5);
								// EDGE : deactivate current layer
								start_layer[current_layer] = 0;
								
								// Display raw results
								/*$display("\n--- Layer 3 Raw Systolic Array Output (C = A × B) ---");
								for (integer i=1; i<LENGTH+1;i++) begin
									$write(" %0d ", raw_result[current_layer][i-1]);
									if (i%3==0) $write("\n");
								end*/
				    end
				    
				    // EDGE : all layers finished
				    status = 0;
				    done = 1'b1;
    			end
    		endcase
    	end
    end
endmodule
