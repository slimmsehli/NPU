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
