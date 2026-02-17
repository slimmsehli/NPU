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
    wire npu_done;


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
			repeat (2) @(posedge clk);
			//@(posedge npu_done);
			repeat (100) @(posedge clk);
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
        	$write(" %0d ", result[2][i-1]);
        	if (i%3==0) $write("\n");
        end
        
        fd = $fopen("../memories/npu_output.hex", "w");
				for (integer i = 0; i < LENGTH; i++) begin
					$fdisplay(fd, "%01h", result[2][i]);
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
				.vpu_valid(vpu_valid), //output [LAYERS-1:0] vpu_valid
				.done(npu_done)
		);  

endmodule
