









// Testbench for Sthe systolic Array 

module top();

    parameter DATA_WIDTH = 8;
    parameter ACC_WIDTH = 32;
    parameter CLK_PERIOD = 10;
    
    reg clk;
    reg rst;
    reg start;
    
    // Input matrices (1D arrays)
    reg [DATA_WIDTH-1:0] matrix_a [0:8];
    reg [DATA_WIDTH-1:0] matrix_b [0:8];
    
    // Controller outputs
    wire [DATA_WIDTH-1:0] west_0, west_1, west_2;
    wire [DATA_WIDTH-1:0] north_0, north_1, north_2;
    wire done;
    
    // Systolic array outputs
    wire [ACC_WIDTH-1:0] result_00, result_01, result_02;
    wire [ACC_WIDTH-1:0] result_10, result_11, result_12;
    wire [ACC_WIDTH-1:0] result_20, result_21, result_22;
    
    // Clock generation
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end
    
    initial begin
      $readmemh("weights.hex", matrix_a);
      $readmemh("inputs.hex", matrix_b);
  end
    
    // Instantiate controller
    systolic_controller #(DATA_WIDTH) controller (
        .clk(clk),
        .rst(rst),
        .start(start),
        .matrix_a(matrix_a),
        .matrix_b(matrix_b),
        .west_0(west_0),
        .west_1(west_1),
        .west_2(west_2),
        .north_0(north_0),
        .north_1(north_1),
        .north_2(north_2),
        .done(done)
    );
    
    // Instantiate systolic array
    systolic_array #(DATA_WIDTH, ACC_WIDTH) array (
        .clk(clk),
        .rst(rst),
        .north_0(north_0),
        .north_1(north_1),
        .north_2(north_2),
        .west_0(west_0),
        .west_1(west_1),
        .west_2(west_2),
        .result_00(result_00),
        .result_01(result_01),
        .result_02(result_02),
        .result_10(result_10),
        .result_11(result_11),
        .result_12(result_12),
        .result_20(result_20),
        .result_21(result_21),
        .result_22(result_22)
    );
    
    // Test procedure
    initial begin
        $display("Starting Systolic Array Matrix Multiplication Test");
        $display("=============================================");
        
        // Initialize
        rst = 1;
        start = 0;
        
        // Initialize Matrix A (3x3)
        // [1 2 3]
        // [4 5 6]
        // [7 8 9]
        //matrix_a[0] = 1; matrix_a[1] = 2; matrix_a[2] = 3;
        //matrix_a[3] = 4; matrix_a[4] = 5; matrix_a[5] = 6;
        //matrix_a[6] = 7; matrix_a[7] = 8; matrix_a[8] = 9;
        
        // Initialize Matrix B (3x3)
        // [9 8 7]
        // [6 5 4]
        // [3 2 1]
        //matrix_b[0] = 9; matrix_b[1] = 8; matrix_b[2] = 7;
        //matrix_b[3] = 6; matrix_b[4] = 5; matrix_b[5] = 4;
        //matrix_b[6] = 3; matrix_b[7] = 2; matrix_b[8] = 1;
        
        $display("\nMatrix A:");
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
        
        // Start computation
        start = 1;
        
        // Wait for computation to complete
        wait(done);
        #(CLK_PERIOD*5);  // Wait a few more cycles for results to stabilize
        
        // Display results
        $display("\nResult Matrix C = A × B:");
        $display("[%0d %0d %0d]", result_00, result_01, result_02);
        $display("[%0d %0d %0d]", result_10, result_11, result_12);
        $display("[%0d %0d %0d]", result_20, result_21, result_22);
        
        $display("\nTest completed!");
        $finish;
    end
    
    // Optional: Monitor signals during simulation
    /*initial begin
        $monitor("Time=%0t | Cycle=%0d | West=[%0d,%0d,%0d] North=[%0d,%0d,%0d]", 
                 $time, controller.cycle_count, 
                 west_0, west_1, west_2, 
                 north_0, north_1, north_2);
    end*/

endmodule
