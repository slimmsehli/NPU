// Vector Processing Unit (VPU)
// Post-processes systolic array outputs with various operations
// Supports: activation functions, pooling, normalization, element-wise ops

module vector_processing_unit #(
    parameter ACC_WIDTH = 32,
    parameter DATA_WIDTH = 8,
    parameter FRAC_BITS = 8  // For fixed-point operations
)(
    input wire clk,
    input wire rst,
    input wire enable,
    
    // Input from systolic array (9 elements for 3x3 result)
    input wire [ACC_WIDTH-1:0] data_in_00,
    input wire [ACC_WIDTH-1:0] data_in_01,
    input wire [ACC_WIDTH-1:0] data_in_02,
    input wire [ACC_WIDTH-1:0] data_in_10,
    input wire [ACC_WIDTH-1:0] data_in_11,
    input wire [ACC_WIDTH-1:0] data_in_12,
    input wire [ACC_WIDTH-1:0] data_in_20,
    input wire [ACC_WIDTH-1:0] data_in_21,
    input wire [ACC_WIDTH-1:0] data_in_22,
    
    // Control signals
    input wire [2:0] operation,     // Select operation type
    input wire [ACC_WIDTH-1:0] bias_value,  // For bias addition
    input wire [ACC_WIDTH-1:0] scale_factor, // For scaling/normalization
    
    // Outputs
    output reg [ACC_WIDTH-1:0] data_out_00,
    output reg [ACC_WIDTH-1:0] data_out_01,
    output reg [ACC_WIDTH-1:0] data_out_02,
    output reg [ACC_WIDTH-1:0] data_out_10,
    output reg [ACC_WIDTH-1:0] data_out_11,
    output reg [ACC_WIDTH-1:0] data_out_12,
    output reg [ACC_WIDTH-1:0] data_out_20,
    output reg [ACC_WIDTH-1:0] data_out_21,
    output reg [ACC_WIDTH-1:0] data_out_22,
    
    output reg valid_out
);

    // Operation codes
    localparam OP_PASSTHROUGH = 3'd0;
    localparam OP_RELU        = 3'd1;
    localparam OP_BIAS_ADD    = 3'd2;
    localparam OP_SCALE       = 3'd3;
    localparam OP_MAX_POOL    = 3'd4;
    localparam OP_AVG_POOL    = 3'd5;
    localparam OP_SIGMOID     = 3'd6;
    localparam OP_TANH        = 3'd7;
    
    // Internal registers for input data
    reg [ACC_WIDTH-1:0] in_array [0:8];
    reg [ACC_WIDTH-1:0] out_array [0:8];
    
    // Helper wires
    wire signed [ACC_WIDTH-1:0] signed_in_00 = $signed(data_in_00);
    wire signed [ACC_WIDTH-1:0] signed_in_01 = $signed(data_in_01);
    wire signed [ACC_WIDTH-1:0] signed_in_02 = $signed(data_in_02);
    wire signed [ACC_WIDTH-1:0] signed_in_10 = $signed(data_in_10);
    wire signed [ACC_WIDTH-1:0] signed_in_11 = $signed(data_in_11);
    wire signed [ACC_WIDTH-1:0] signed_in_12 = $signed(data_in_12);
    wire signed [ACC_WIDTH-1:0] signed_in_20 = $signed(data_in_20);
    wire signed [ACC_WIDTH-1:0] signed_in_21 = $signed(data_in_21);
    wire signed [ACC_WIDTH-1:0] signed_in_22 = $signed(data_in_22);
    
    integer i;
    
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            data_out_00 <= 0;
            data_out_01 <= 0;
            data_out_02 <= 0;
            data_out_10 <= 0;
            data_out_11 <= 0;
            data_out_12 <= 0;
            data_out_20 <= 0;
            data_out_21 <= 0;
            data_out_22 <= 0;
            valid_out <= 0;
        end else if (enable) begin
            // Load input array
            in_array[0] = data_in_00;
            in_array[1] = data_in_01;
            in_array[2] = data_in_02;
            in_array[3] = data_in_10;
            in_array[4] = data_in_11;
            in_array[5] = data_in_12;
            in_array[6] = data_in_20;
            in_array[7] = data_in_21;
            in_array[8] = data_in_22;
            
            case (operation)
                OP_PASSTHROUGH: begin
                    // Direct pass-through
                    for (i = 0; i < 9; i = i + 1) begin
                        out_array[i] = in_array[i];
                    end
                end
                
                OP_RELU: begin
                    // ReLU activation: max(0, x)
                    out_array[0] = (signed_in_00 < 0) ? 0 : data_in_00;
                    out_array[1] = (signed_in_01 < 0) ? 0 : data_in_01;
                    out_array[2] = (signed_in_02 < 0) ? 0 : data_in_02;
                    out_array[3] = (signed_in_10 < 0) ? 0 : data_in_10;
                    out_array[4] = (signed_in_11 < 0) ? 0 : data_in_11;
                    out_array[5] = (signed_in_12 < 0) ? 0 : data_in_12;
                    out_array[6] = (signed_in_20 < 0) ? 0 : data_in_20;
                    out_array[7] = (signed_in_21 < 0) ? 0 : data_in_21;
                    out_array[8] = (signed_in_22 < 0) ? 0 : data_in_22;
                end
                
                OP_BIAS_ADD: begin
                    // Add bias to all elements
                    for (i = 0; i < 9; i = i + 1) begin
                        out_array[i] = in_array[i] + bias_value;
                    end
                end
                
                OP_SCALE: begin
                    // Multiply by scale factor (with right shift for fixed-point)
                    for (i = 0; i < 9; i = i + 1) begin
                        out_array[i] = (in_array[i] * scale_factor) >> FRAC_BITS;
                    end
                end
                
                OP_MAX_POOL: begin
                    // 2x2 Max pooling (4 outputs, rest zeros)
                    // Pool regions: [0,1,3,4], [1,2,4,5], [3,4,6,7], [4,5,7,8]
                    out_array[0] = max4(in_array[0], in_array[1], in_array[3], in_array[4]);
                    out_array[1] = max4(in_array[1], in_array[2], in_array[4], in_array[5]);
                    out_array[2] = 0;
                    out_array[3] = max4(in_array[3], in_array[4], in_array[6], in_array[7]);
                    out_array[4] = max4(in_array[4], in_array[5], in_array[7], in_array[8]);
                    out_array[5] = 0;
                    out_array[6] = 0;
                    out_array[7] = 0;
                    out_array[8] = 0;
                end
                
                OP_AVG_POOL: begin
                    // 2x2 Average pooling (4 outputs, rest zeros)
                    out_array[0] = (in_array[0] + in_array[1] + in_array[3] + in_array[4]) >> 2;
                    out_array[1] = (in_array[1] + in_array[2] + in_array[4] + in_array[5]) >> 2;
                    out_array[2] = 0;
                    out_array[3] = (in_array[3] + in_array[4] + in_array[6] + in_array[7]) >> 2;
                    out_array[4] = (in_array[4] + in_array[5] + in_array[7] + in_array[8]) >> 2;
                    out_array[5] = 0;
                    out_array[6] = 0;
                    out_array[7] = 0;
                    out_array[8] = 0;
                end
                
                OP_SIGMOID: begin
                    // Simplified sigmoid approximation using lookup/piecewise
                    for (i = 0; i < 9; i = i + 1) begin
                        out_array[i] = sigmoid_approx(in_array[i]);
                    end
                end
                
                OP_TANH: begin
                    // Simplified tanh approximation
                    for (i = 0; i < 9; i = i + 1) begin
                        out_array[i] = tanh_approx(in_array[i]);
                    end
                end
                
                default: begin
                    // Default to pass-through
                    for (i = 0; i < 9; i = i + 1) begin
                        out_array[i] = in_array[i];
                    end
                end
            endcase
            
            // Write outputs
            data_out_00 <= out_array[0];
            data_out_01 <= out_array[1];
            data_out_02 <= out_array[2];
            data_out_10 <= out_array[3];
            data_out_11 <= out_array[4];
            data_out_12 <= out_array[5];
            data_out_20 <= out_array[6];
            data_out_21 <= out_array[7];
            data_out_22 <= out_array[8];
            
            valid_out <= 1;
        end else begin
            valid_out <= 0;
        end
    end
    
    // Helper function: max of 4 values
    function [ACC_WIDTH-1:0] max4;
        input [ACC_WIDTH-1:0] a, b, c, d;
        reg [ACC_WIDTH-1:0] max_ab, max_cd;
        begin
            max_ab = (a > b) ? a : b;
            max_cd = (c > d) ? c : d;
            max4 = (max_ab > max_cd) ? max_ab : max_cd;
        end
    endfunction
    
    // Simplified sigmoid approximation (piecewise linear)
    function [ACC_WIDTH-1:0] sigmoid_approx;
        input signed [ACC_WIDTH-1:0] x;
        reg signed [ACC_WIDTH-1:0] result;
        begin
            if (x < -4 << FRAC_BITS)
                result = 0;
            else if (x > 4 << FRAC_BITS)
                result = 1 << FRAC_BITS;
            else
                // Linear approximation: y = 0.5 + x/8
                result = (1 << (FRAC_BITS-1)) + (x >> 3);
            sigmoid_approx = result;
        end
    endfunction
    
    // Simplified tanh approximation (piecewise linear)
    function [ACC_WIDTH-1:0] tanh_approx;
        input signed [ACC_WIDTH-1:0] x;
        reg signed [ACC_WIDTH-1:0] result;
        begin
            if (x < -2 << FRAC_BITS)
                result = -(1 << FRAC_BITS);
            else if (x > 2 << FRAC_BITS)
                result = 1 << FRAC_BITS;
            else
                // Linear approximation: y = x/2
                result = x >> 1;
            tanh_approx = result;
        end
    endfunction

endmodule
