// Systolic Array Input Controller
// Loads 3x3 matrices from 1D arrays into systolic array inputs
// Matrix A enters from the WEST, Matrix B enters from the NORTH

module systolic_controller #(
    parameter DATA_WIDTH = 8
)(
    input wire clk,
    input wire rst,
    input wire start,
    
    // Matrix A (1D array, 9 elements) - enters from WEST
    input wire [DATA_WIDTH-1:0] matrix_a [0:8],
    
    // Matrix B (1D array, 9 elements) - enters from NORTH  
    input wire [DATA_WIDTH-1:0] matrix_b [0:8],
    
    // West inputs (rows of A, staggered)
    output reg [DATA_WIDTH-1:0] west_0,
    output reg [DATA_WIDTH-1:0] west_1,
    output reg [DATA_WIDTH-1:0] west_2,
    
    // North inputs (columns of B, staggered)
    output reg [DATA_WIDTH-1:0] north_0,
    output reg [DATA_WIDTH-1:0] north_1,
    output reg [DATA_WIDTH-1:0] north_2,
    
    output reg done
);

    reg [3:0] cycle_count;
    
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            cycle_count <= 0;
            west_0 <= 0;
            west_1 <= 0;
            west_2 <= 0;
            north_0 <= 0;
            north_1 <= 0;
            north_2 <= 0;
            done <= 0;
        end else if (start) begin
            cycle_count <= cycle_count + 1;
            
            // West inputs (Matrix A by rows, staggered)
            case (cycle_count)
                0: begin
                    west_0 <= matrix_a[0]; // A[0][0]
                    west_1 <= 0;           // idle
                    west_2 <= 0;           // idle
                end
                1: begin
                    west_0 <= matrix_a[1]; // A[0][1]
                    west_1 <= matrix_a[3]; // A[1][0]
                    west_2 <= 0;           // idle
                end
                2: begin
                    west_0 <= matrix_a[2]; // A[0][2]
                    west_1 <= matrix_a[4]; // A[1][1]
                    west_2 <= matrix_a[6]; // A[2][0]
                end
                3: begin
                    west_0 <= 0;           // idle
                    west_1 <= matrix_a[5]; // A[1][2]
                    west_2 <= matrix_a[7]; // A[2][1]
                end
                4: begin
                    west_0 <= 0;           // idle
                    west_1 <= 0;           // idle
                    west_2 <= matrix_a[8]; // A[2][2]
                end
                default: begin
                    west_0 <= 0;
                    west_1 <= 0;
                    west_2 <= 0;
                end
            endcase
            
            // North inputs (Matrix B by columns, staggered)
            case (cycle_count)
                0: begin
                    north_0 <= matrix_b[0]; // B[0][0]
                    north_1 <= 0;           // idle
                    north_2 <= 0;           // idle
                end
                1: begin
                    north_0 <= matrix_b[3]; // B[1][0]
                    north_1 <= matrix_b[1]; // B[0][1]
                    north_2 <= 0;           // idle
                end
                2: begin
                    north_0 <= matrix_b[6]; // B[2][0]
                    north_1 <= matrix_b[4]; // B[1][1]
                    north_2 <= matrix_b[2]; // B[0][2]
                end
                3: begin
                    north_0 <= 0;           // idle
                    north_1 <= matrix_b[7]; // B[2][1]
                    north_2 <= matrix_b[5]; // B[1][2]
                end
                4: begin
                    north_0 <= 0;           // idle
                    north_1 <= 0;           // idle
                    north_2 <= matrix_b[8]; // B[2][2]
                end
                default: begin
                    north_0 <= 0;
                    north_1 <= 0;
                    north_2 <= 0;
                end
            endcase
            
            // Signal done after all inputs have been fed
            if (cycle_count >= 8) begin
                done <= 1;
            end
        end
    end

endmodule



// Processing Element for Systolic Array
// Performs multiply-accumulate operation
// Passes data from north to south, from west to east

module processing_element #(
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH = 32
)(
    input wire clk,
    input wire rst,
    
    // Data inputs
    input wire [DATA_WIDTH-1:0] data_in_north,
    input wire [DATA_WIDTH-1:0] data_in_west,
    
    // Data outputs (passed through)
    output reg [DATA_WIDTH-1:0] data_out_south,
    output reg [DATA_WIDTH-1:0] data_out_east,
    
    // Accumulated result
    output reg [ACC_WIDTH-1:0] acc_out
);

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            data_out_south <= 0;
            data_out_east <= 0;
            acc_out <= 0;
        end else begin
            // Pass through data
            data_out_south <= data_in_north;
            data_out_east <= data_in_west;
            
            // Accumulate: acc = acc + (north * west)
            acc_out <= acc_out + (data_in_north * data_in_west);
        end
    end

endmodule








// 3x3 Systolic Array for Matrix Multiplication
// Computes C = A × B where A and B are 3x3 matrices

module systolic_array #(
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH = 32
)(
    input wire clk,
    input wire rst,
    
    // North inputs (columns of matrix B)
    input wire [DATA_WIDTH-1:0] north_0,
    input wire [DATA_WIDTH-1:0] north_1,
    input wire [DATA_WIDTH-1:0] north_2,
    
    // West inputs (rows of matrix A)
    input wire [DATA_WIDTH-1:0] west_0,
    input wire [DATA_WIDTH-1:0] west_1,
    input wire [DATA_WIDTH-1:0] west_2,
    
    // Output matrix C (9 elements)
    output wire [ACC_WIDTH-1:0] result_00,
    output wire [ACC_WIDTH-1:0] result_01,
    output wire [ACC_WIDTH-1:0] result_02,
    output wire [ACC_WIDTH-1:0] result_10,
    output wire [ACC_WIDTH-1:0] result_11,
    output wire [ACC_WIDTH-1:0] result_12,
    output wire [ACC_WIDTH-1:0] result_20,
    output wire [ACC_WIDTH-1:0] result_21,
    output wire [ACC_WIDTH-1:0] result_22
);

    // Internal connections between PEs
    // Vertical connections (north to south)
    wire [DATA_WIDTH-1:0] v_01, v_02;
    wire [DATA_WIDTH-1:0] v_11, v_12;
    wire [DATA_WIDTH-1:0] v_21, v_22;
    
    // Horizontal connections (west to east)
    wire [DATA_WIDTH-1:0] h_10, h_20;
    wire [DATA_WIDTH-1:0] h_11, h_21;
    wire [DATA_WIDTH-1:0] h_12, h_22;
    
    // Row 0 PEs
    processing_element #(DATA_WIDTH, ACC_WIDTH) pe_00 (
        .clk(clk),
        .rst(rst),
        .data_in_north(north_0),
        .data_in_west(west_0),
        .data_out_south(v_01),
        .data_out_east(h_10),
        .acc_out(result_00)
    );
    
    processing_element #(DATA_WIDTH, ACC_WIDTH) pe_01 (
        .clk(clk),
        .rst(rst),
        .data_in_north(north_1),
        .data_in_west(h_10),
        .data_out_south(v_11),
        .data_out_east(h_20),
        .acc_out(result_01)
    );
    
    processing_element #(DATA_WIDTH, ACC_WIDTH) pe_02 (
        .clk(clk),
        .rst(rst),
        .data_in_north(north_2),
        .data_in_west(h_20),
        .data_out_south(v_21),
        .data_out_east(),  // unused
        .acc_out(result_02)
    );
    
    // Row 1 PEs
    processing_element #(DATA_WIDTH, ACC_WIDTH) pe_10 (
        .clk(clk),
        .rst(rst),
        .data_in_north(v_01),
        .data_in_west(west_1),
        .data_out_south(v_02),
        .data_out_east(h_11),
        .acc_out(result_10)
    );
    
    processing_element #(DATA_WIDTH, ACC_WIDTH) pe_11 (
        .clk(clk),
        .rst(rst),
        .data_in_north(v_11),
        .data_in_west(h_11),
        .data_out_south(v_12),
        .data_out_east(h_21),
        .acc_out(result_11)
    );
    
    processing_element #(DATA_WIDTH, ACC_WIDTH) pe_12 (
        .clk(clk),
        .rst(rst),
        .data_in_north(v_21),
        .data_in_west(h_21),
        .data_out_south(v_22),
        .data_out_east(),  // unused
        .acc_out(result_12)
    );
    
    // Row 2 PEs
    processing_element #(DATA_WIDTH, ACC_WIDTH) pe_20 (
        .clk(clk),
        .rst(rst),
        .data_in_north(v_02),
        .data_in_west(west_2),
        .data_out_south(),  // unused
        .data_out_east(h_12),
        .acc_out(result_20)
    );
    
    processing_element #(DATA_WIDTH, ACC_WIDTH) pe_21 (
        .clk(clk),
        .rst(rst),
        .data_in_north(v_12),
        .data_in_west(h_12),
        .data_out_south(),  // unused
        .data_out_east(h_22),
        .acc_out(result_21)
    );
    
    processing_element #(DATA_WIDTH, ACC_WIDTH) pe_22 (
        .clk(clk),
        .rst(rst),
        .data_in_north(v_22),
        .data_in_west(h_22),
        .data_out_south(),  // unused
        .data_out_east(),   // unused
        .acc_out(result_22)
    );

endmodule


// Testbench for Systolic Array Matrix Multiplication


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
