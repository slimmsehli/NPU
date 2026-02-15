// Neural Processing Unit (NPU) - Top Module
// Parameterizable multi-layer neural network accelerator
// Uses systolic array for matrix multiplication and VPU for post-processing

module npu_top #(
    parameter NUM_LAYERS = 3,           // Number of layers in the network
    parameter DATA_WIDTH = 8,           // Input data width
    parameter ACC_WIDTH = 32,           // Accumulator width
    parameter FRAC_BITS = 8,            // Fixed-point fractional bits
    parameter MATRIX_SIZE = 9           // 3x3 matrices (9 elements)
)(
    input wire clk,
    input wire rst,
    input wire start,                   // Start inference
    
    // Input data (initial input to first layer)
    input wire [DATA_WIDTH-1:0] input_data [0:MATRIX_SIZE-1],
    
    // Weight matrices for each layer (stored externally, loaded per layer)
    input wire [DATA_WIDTH-1:0] weights [0:MATRIX_SIZE-1],
    
    // Bias values for each layer
    input wire [ACC_WIDTH-1:0] bias_layer [0:NUM_LAYERS-1],
    
    // VPU operations for each layer
    input wire [2:0] vpu_ops_layer [0:NUM_LAYERS-1],
    
    // Scale factors for normalization (if needed)
    input wire [ACC_WIDTH-1:0] scale_layer [0:NUM_LAYERS-1],
    
    // Final output (after all layers processed)
    output reg [ACC_WIDTH-1:0] output_data [0:MATRIX_SIZE-1],
    
    // Status signals
    output reg done,
    output reg [7:0] current_layer,     // Which layer is being processed
    output reg [3:0] state              // Current FSM state
);

    // FSM States
    localparam IDLE           = 4'd0;
    localparam LOAD_WEIGHTS   = 4'd1;
    localparam START_MATMUL   = 4'd2;
    localparam WAIT_MATMUL    = 4'd3;
    localparam APPLY_BIAS     = 4'd4;
    localparam APPLY_ACTIVATION = 4'd5;
    localparam STORE_RESULT   = 4'd6;
    localparam NEXT_LAYER     = 4'd7;
    localparam COMPLETE       = 4'd8;
    
    // VPU operation codes
    localparam OP_PASSTHROUGH = 3'd0;
    localparam OP_RELU        = 3'd1;
    localparam OP_BIAS_ADD    = 3'd2;
    localparam OP_SCALE       = 3'd3;
    localparam OP_MAX_POOL    = 3'd4;
    localparam OP_AVG_POOL    = 3'd5;
    localparam OP_SIGMOID     = 3'd6;
    localparam OP_TANH        = 3'd7;
    
    // Internal registers
    reg [7:0] layer_counter;
    reg [7:0] cycle_counter;
    
    // Systolic array controller signals
    reg systolic_start;
    wire systolic_done;
    reg [DATA_WIDTH-1:0] matrix_a_input [0:MATRIX_SIZE-1];
    reg [DATA_WIDTH-1:0] matrix_b_input [0:MATRIX_SIZE-1];
    
    // Systolic array outputs (raw)
    wire [ACC_WIDTH-1:0] systolic_out_00, systolic_out_01, systolic_out_02;
    wire [ACC_WIDTH-1:0] systolic_out_10, systolic_out_11, systolic_out_12;
    wire [ACC_WIDTH-1:0] systolic_out_20, systolic_out_21, systolic_out_22;
    
    // VPU signals
    reg vpu_enable;
    reg [2:0] vpu_operation;
    reg [ACC_WIDTH-1:0] vpu_bias;
    reg [ACC_WIDTH-1:0] vpu_scale;
    wire vpu_valid;
    
    // VPU outputs
    wire [ACC_WIDTH-1:0] vpu_out_00, vpu_out_01, vpu_out_02;
    wire [ACC_WIDTH-1:0] vpu_out_10, vpu_out_11, vpu_out_12;
    wire [ACC_WIDTH-1:0] vpu_out_20, vpu_out_21, vpu_out_22;
    
    // Layer buffers - stores intermediate results between layers
    reg [ACC_WIDTH-1:0] layer_buffer [0:MATRIX_SIZE-1];
    
    // Helper array for VPU output collection
    wire [ACC_WIDTH-1:0] vpu_output_array [0:8];
    assign vpu_output_array[0] = vpu_out_00;
    assign vpu_output_array[1] = vpu_out_01;
    assign vpu_output_array[2] = vpu_out_02;
    assign vpu_output_array[3] = vpu_out_10;
    assign vpu_output_array[4] = vpu_out_11;
    assign vpu_output_array[5] = vpu_out_12;
    assign vpu_output_array[6] = vpu_out_20;
    assign vpu_output_array[7] = vpu_out_21;
    assign vpu_output_array[8] = vpu_out_22;
    
    integer i;
    
    // Instantiate integrated systolic array with VPU
    systolic_with_vpu #(
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .FRAC_BITS(FRAC_BITS)
    ) compute_engine (
        .clk(clk),
        .rst(rst),
        .start(systolic_start),
        .matrix_a(matrix_a_input),
        .matrix_b(matrix_b_input),
        .vpu_enable(vpu_enable),
        .vpu_operation(vpu_operation),
        .bias_value(vpu_bias),
        .scale_factor(vpu_scale),
        .raw_result_00(systolic_out_00),
        .raw_result_01(systolic_out_01),
        .raw_result_02(systolic_out_02),
        .raw_result_10(systolic_out_10),
        .raw_result_11(systolic_out_11),
        .raw_result_12(systolic_out_12),
        .raw_result_20(systolic_out_20),
        .raw_result_21(systolic_out_21),
        .raw_result_22(systolic_out_22),
        .result_00(vpu_out_00),
        .result_01(vpu_out_01),
        .result_02(vpu_out_02),
        .result_10(vpu_out_10),
        .result_11(vpu_out_11),
        .result_12(vpu_out_12),
        .result_20(vpu_out_20),
        .result_21(vpu_out_21),
        .result_22(vpu_out_22),
        .controller_done(systolic_done),
        .vpu_valid(vpu_valid)
    );
    
    // Main FSM
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= IDLE;
            layer_counter <= 0;
            cycle_counter <= 0;
            systolic_start <= 0;
            vpu_enable <= 0;
            vpu_operation <= OP_PASSTHROUGH;
            vpu_bias <= 0;
            vpu_scale <= 0;
            done <= 0;
            current_layer <= 0;
            
            // Clear buffers
            for (i = 0; i < MATRIX_SIZE; i = i + 1) begin
                layer_buffer[i] <= 0;
                matrix_a_input[i] <= 0;
                matrix_b_input[i] <= 0;
                output_data[i] <= 0;
            end
            
        end else begin
            case (state)
                
                IDLE: begin
                    done <= 0;
                    systolic_start <= 0;
                    vpu_enable <= 0;
                    
                    if (start) begin
                        layer_counter <= 0;
                        current_layer <= 0;
                        state <= LOAD_WEIGHTS;
                        
                        // Load initial input data into layer buffer
                        for (i = 0; i < MATRIX_SIZE; i = i + 1) begin
                            layer_buffer[i] <= {{(ACC_WIDTH-DATA_WIDTH){1'b0}}, input_data[i]};
                        end
                    end
                end
                
                LOAD_WEIGHTS: begin
                    // Load input from previous layer (or initial input)
                    // For first layer, data is already in layer_buffer
                    // For subsequent layers, convert ACC_WIDTH back to DATA_WIDTH
                    for (i = 0; i < MATRIX_SIZE; i = i + 1) begin
                        matrix_a_input[i] <= layer_buffer[i][DATA_WIDTH-1:0];
                        matrix_b_input[i] <= weights[i];
                    end
                    
                    cycle_counter <= 0;
                    state <= START_MATMUL;
                end
                
                START_MATMUL: begin
                    systolic_start <= 1;
                    state <= WAIT_MATMUL;
                end
                
                WAIT_MATMUL: begin
                    cycle_counter <= cycle_counter + 1;
                    
                    // Wait for systolic array to complete
                    // Typically takes ~9-10 cycles for 3x3
                    if (systolic_done && cycle_counter >= 10) begin
                        systolic_start <= 0;
                        state <= APPLY_BIAS;
                    end
                end
                
                APPLY_BIAS: begin
                    // Apply bias using VPU
                    vpu_operation <= OP_BIAS_ADD;
                    vpu_bias <= bias_layer[layer_counter];
                    vpu_enable <= 1;
                    cycle_counter <= 0;
                    state <= APPLY_ACTIVATION;
                end
                
                APPLY_ACTIVATION: begin
                    cycle_counter <= cycle_counter + 1;
                    
                    // Wait for bias VPU operation to complete
                    if (vpu_valid && cycle_counter >= 1) begin
                        // Now apply the activation function for this layer
                        vpu_operation <= vpu_ops_layer[layer_counter];
                        vpu_scale <= scale_layer[layer_counter];
                        vpu_enable <= 1;
                        cycle_counter <= 0;
                        state <= STORE_RESULT;
                    end else begin
                        vpu_enable <= 0;
                    end
                end
                
                STORE_RESULT: begin
                    cycle_counter <= cycle_counter + 1;
                    
                    // Wait for VPU to produce valid output, then capture it
                    if (vpu_valid && cycle_counter >= 1) begin
                        // Store VPU output into layer buffer for next layer
                        for (i = 0; i < MATRIX_SIZE; i = i + 1) begin
                            layer_buffer[i] <= vpu_output_array[i];
                        end
                        vpu_enable <= 0;
                        state <= NEXT_LAYER;
                    end
                end
                
                NEXT_LAYER: begin
                    layer_counter <= layer_counter + 1;
                    current_layer <= layer_counter + 1;
                    
                    // Check if all layers are processed
                    if (layer_counter >= NUM_LAYERS - 1) begin
                        state <= COMPLETE;
                    end else begin
                        state <= LOAD_WEIGHTS;
                    end
                end
                
                COMPLETE: begin
                    // Copy final result to output
                    for (i = 0; i < MATRIX_SIZE; i = i + 1) begin
                        output_data[i] <= layer_buffer[i];
                    end
                    done <= 1;
                    state <= IDLE;
                end
                
                default: state <= IDLE;
            endcase
        end
    end

endmodule
