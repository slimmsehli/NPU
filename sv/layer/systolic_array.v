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


// Integrated System: Systolic Array + Vector Processing Unit
// Complete matrix multiplication with post-processing

module systolic_with_vpu #(
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH = 32,
    parameter FRAC_BITS = 8
)(
    input wire clk,
    input wire rst,
    input wire start,
    
    // Matrix inputs (1D arrays)
    input wire [DATA_WIDTH-1:0] matrix_a [0:8],
    input wire [DATA_WIDTH-1:0] matrix_b [0:8],
    
    // VPU control
    input wire vpu_enable,
    input wire [2:0] vpu_operation,
    input wire [ACC_WIDTH-1:0] bias_value,
    input wire [ACC_WIDTH-1:0] scale_factor,
    
    // Raw systolic array outputs (before VPU)
    output wire [ACC_WIDTH-1:0] raw_result_00,
    output wire [ACC_WIDTH-1:0] raw_result_01,
    output wire [ACC_WIDTH-1:0] raw_result_02,
    output wire [ACC_WIDTH-1:0] raw_result_10,
    output wire [ACC_WIDTH-1:0] raw_result_11,
    output wire [ACC_WIDTH-1:0] raw_result_12,
    output wire [ACC_WIDTH-1:0] raw_result_20,
    output wire [ACC_WIDTH-1:0] raw_result_21,
    output wire [ACC_WIDTH-1:0] raw_result_22,
    
    // Processed outputs (after VPU)
    output wire [ACC_WIDTH-1:0] result_00,
    output wire [ACC_WIDTH-1:0] result_01,
    output wire [ACC_WIDTH-1:0] result_02,
    output wire [ACC_WIDTH-1:0] result_10,
    output wire [ACC_WIDTH-1:0] result_11,
    output wire [ACC_WIDTH-1:0] result_12,
    output wire [ACC_WIDTH-1:0] result_20,
    output wire [ACC_WIDTH-1:0] result_21,
    output wire [ACC_WIDTH-1:0] result_22,
    
    output wire controller_done,
    output wire vpu_valid
);

    // Internal signals
    wire [DATA_WIDTH-1:0] west_0, west_1, west_2;
    wire [DATA_WIDTH-1:0] north_0, north_1, north_2;
    
    // Instantiate controller
    systolic_controller #(
        .DATA_WIDTH(DATA_WIDTH)
    ) controller (
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
        .done(controller_done)
    );
    
    // Instantiate systolic array
    systolic_array #(
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(ACC_WIDTH)
    ) array (
        .clk(clk),
        .rst(rst),
        .north_0(north_0),
        .north_1(north_1),
        .north_2(north_2),
        .west_0(west_0),
        .west_1(west_1),
        .west_2(west_2),
        .result_00(raw_result_00),
        .result_01(raw_result_01),
        .result_02(raw_result_02),
        .result_10(raw_result_10),
        .result_11(raw_result_11),
        .result_12(raw_result_12),
        .result_20(raw_result_20),
        .result_21(raw_result_21),
        .result_22(raw_result_22)
    );
    
    // Instantiate Vector Processing Unit
    vector_processing_unit #(
        .ACC_WIDTH(ACC_WIDTH),
        .DATA_WIDTH(DATA_WIDTH),
        .FRAC_BITS(FRAC_BITS)
    ) vpu (
        .clk(clk),
        .rst(rst),
        .enable(vpu_enable),
        .data_in_00(raw_result_00),
        .data_in_01(raw_result_01),
        .data_in_02(raw_result_02),
        .data_in_10(raw_result_10),
        .data_in_11(raw_result_11),
        .data_in_12(raw_result_12),
        .data_in_20(raw_result_20),
        .data_in_21(raw_result_21),
        .data_in_22(raw_result_22),
        .operation(vpu_operation),
        .bias_value(bias_value),
        .scale_factor(scale_factor),
        .data_out_00(result_00),
        .data_out_01(result_01),
        .data_out_02(result_02),
        .data_out_10(result_10),
        .data_out_11(result_11),
        .data_out_12(result_12),
        .data_out_20(result_20),
        .data_out_21(result_21),
        .data_out_22(result_22),
        .valid_out(vpu_valid)
    );

endmodule
