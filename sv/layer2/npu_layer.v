// Integrated System: Systolic Array + Vector Processing Unit
// Complete matrix multiplication with post-processing

module npu_layer #(
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH = 32,
    parameter FRAC_BITS = 8,
    parameter LENGTH = 9
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
    output wire [ACC_WIDTH-1:0] raw_result [0:LENGTH-1],
    
    // Processed outputs (after VPU)
    output wire [ACC_WIDTH-1:0] result [0:LENGTH-1],
    
    output wire controller_done,
    output wire vpu_valid
);
		reg vpu_enable_internal, en_vpu;
		always @(posedge start) begin
			vpu_enable_internal = vpu_enable;
		end
		always @(posedge controller_done) begin
			en_vpu = vpu_enable_internal;
		end
		always @(negedge rst) begin
			vpu_enable_internal = 1'b0;
		end
		

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
        .result_00(raw_result[0]),
        .result_01(raw_result[1]),
        .result_02(raw_result[2]),
        .result_10(raw_result[3]),
        .result_11(raw_result[4]),
        .result_12(raw_result[5]),
        .result_20(raw_result[6]),
        .result_21(raw_result[7]),
        .result_22(raw_result[8])
    );
    
    // Instantiate Vector Processing Unit
    vector_processing_unit #(
        .ACC_WIDTH(ACC_WIDTH),
        .DATA_WIDTH(DATA_WIDTH),
        .FRAC_BITS(FRAC_BITS),
        .LENGTH(LENGTH)
    ) vpu (
        .clk(clk),
        .rst(rst),
        .enable(en_vpu),
        .data_in(raw_result),
        .operation(vpu_operation),
        .bias_value(bias_value),
        .scale_factor(scale_factor),
        .data_out(result),
        .valid_out(vpu_valid)
    );

endmodule
