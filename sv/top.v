// Testbench for NPU Top Module
// Tests a 3-layer neural network

module top();

    parameter NUM_LAYERS = 3;
    parameter DATA_WIDTH = 8;
    parameter ACC_WIDTH = 32;
    parameter FRAC_BITS = 8;
    parameter MATRIX_SIZE = 9;
    parameter CLK_PERIOD = 10;
    
    reg clk;
    reg rst;
    reg start;
    
    // Input data
    reg [DATA_WIDTH-1:0] input_data [0:MATRIX_SIZE-1];
    
    // Weights for current layer being processed
    reg [DATA_WIDTH-1:0] weights [0:MATRIX_SIZE-1];
    
    // Per-layer configuration
    reg [ACC_WIDTH-1:0] bias_layer [0:NUM_LAYERS-1];
    reg [2:0] vpu_ops_layer [0:NUM_LAYERS-1];
    reg [ACC_WIDTH-1:0] scale_layer [0:NUM_LAYERS-1];
    
    // Outputs
    wire [ACC_WIDTH-1:0] output_data [0:MATRIX_SIZE-1];
    wire done;
    wire [7:0] current_layer;
    wire [3:0] state;
    
    // Weight storage for all layers
    reg [DATA_WIDTH-1:0] layer1_weights [0:MATRIX_SIZE-1];
    reg [DATA_WIDTH-1:0] layer2_weights [0:MATRIX_SIZE-1];
    reg [DATA_WIDTH-1:0] layer3_weights [0:MATRIX_SIZE-1];
    
    // VPU operation codes
    localparam OP_PASSTHROUGH = 3'd0;
    localparam OP_RELU        = 3'd1;
    localparam OP_BIAS_ADD    = 3'd2;
    localparam OP_SCALE       = 3'd3;
    localparam OP_MAX_POOL    = 3'd4;
    localparam OP_AVG_POOL    = 3'd5;
    localparam OP_SIGMOID     = 3'd6;
    localparam OP_TANH        = 3'd7;
    
    integer i;
    
    // Clock generation
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end
    
    /// load memory
    initial begin
      $readmemh("/global/scsg_eu_verif/slim/test_py/NPU/memories/weights_L0.hex", layer1_weights);
      $readmemh("/global/scsg_eu_verif/slim/test_py/NPU/memories/weights_L1.hex", layer2_weights);
      $readmemh("/global/scsg_eu_verif/slim/test_py/NPU/memories/weights_L2.hex", layer3_weights);
      $readmemh("/global/scsg_eu_verif/slim/test_py/NPU/memories/inputs.hex", input_data);
      bias_layer = {0, 0, 0};
      vpu_ops_layer = {OP_RELU, OP_RELU, OP_RELU};
      scale_layer = {1, 1, 1};
  	end
    
    // Weight update logic - simulates weight memory
    always @(posedge clk) begin
        case (current_layer)
            0: begin
                for (i = 0; i < MATRIX_SIZE; i = i + 1) begin
                    weights[i] = layer1_weights[i];
                end
            end
            1: begin
                for (i = 0; i < MATRIX_SIZE; i = i + 1) begin
                    weights[i] = layer2_weights[i];
                end
            end
            2: begin
                for (i = 0; i < MATRIX_SIZE; i = i + 1) begin
                    weights[i] = layer3_weights[i];
                end
            end
            default: begin
                for (i = 0; i < MATRIX_SIZE; i = i + 1) begin
                    weights[i] = layer1_weights[i];
                end
            end
        endcase
    end
    
    // Instantiate NPU
    npu_top #(
        .NUM_LAYERS(NUM_LAYERS),
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .FRAC_BITS(FRAC_BITS),
        .MATRIX_SIZE(MATRIX_SIZE)
    ) dut (
        .clk(clk),
        .rst(rst),
        .start(start),
        .input_data(input_data),
        .weights(weights),
        .bias_layer(bias_layer),
        .vpu_ops_layer(vpu_ops_layer),
        .scale_layer(scale_layer),
        .output_data(output_data),
        .done(done),
        .current_layer(current_layer),
        .state(state)
    );
    
    // Test procedure
    initial begin
        $display("=========================================================");
        $display("NPU Top Module Test - 3-Layer Neural Network");
        $display("=========================================================\n");
        
        // Initialize
        rst = 1;
        start = 0;
        
        // Initialize input data (3x3 input)
        // Simple test pattern
        //input_data[0] = 1; input_data[1] = 0; input_data[2] = 1;
        //input_data[3] = 0; input_data[4] = 1; input_data[5] = 0;
        //input_data[6] = 1; input_data[7] = 0; input_data[8] = 1;
        
        $display("Input Data (3x3):");
        $display("[%0d %0d %0d]", input_data[0], input_data[1], input_data[2]);
        $display("[%0d %0d %0d]", input_data[3], input_data[4], input_data[5]);
        $display("[%0d %0d %0d]", input_data[6], input_data[7], input_data[8]);
        
        // Configure Layer 1: Hidden Layer with ReLU
        $display("\n--- Layer 1 Configuration ---");
        $display("Type: Hidden Layer (MatMul + Bias + ReLU)");
        //layer1_weights[0] = 2; layer1_weights[1] = 1; layer1_weights[2] = 2;
        //layer1_weights[3] = 1; layer1_weights[4] = 2; layer1_weights[5] = 1;
        //layer1_weights[6] = 2; layer1_weights[7] = 1; layer1_weights[8] = 2;
        
        $display("Weights:");
        $display("[%0d %0d %0d]", layer1_weights[0], layer1_weights[1], layer1_weights[2]);
        $display("[%0d %0d %0d]", layer1_weights[3], layer1_weights[4], layer1_weights[5]);
        $display("[%0d %0d %0d]", layer1_weights[6], layer1_weights[7], layer1_weights[8]);
        
        //bias_layer[0] = 0;
        //vpu_ops_layer[0] = OP_RELU;
        //scale_layer[0] = 256; // 1.0 in fixed point
        $display("Bias: %0d", bias_layer[0]);
        $display("Activation: ReLU");
        
        // Configure Layer 2: Hidden Layer with ReLU
        $display("\n--- Layer 2 Configuration ---");
        $display("Type: Hidden Layer (MatMul + Bias + ReLU)");
        //layer2_weights[0] = 1; layer2_weights[1] = 1; layer2_weights[2] = 1;
        //layer2_weights[3] = 1; layer2_weights[4] = 1; layer2_weights[5] = 1;
        //layer2_weights[6] = 1; layer2_weights[7] = 1; layer2_weights[8] = 1;
        
        $display("Weights:");
        $display("[%0d %0d %0d]", layer2_weights[0], layer2_weights[1], layer2_weights[2]);
        $display("[%0d %0d %0d]", layer2_weights[3], layer2_weights[4], layer2_weights[5]);
        $display("[%0d %0d %0d]", layer2_weights[6], layer2_weights[7], layer2_weights[8]);
        
        //bias_layer[1] = 0;
        //vpu_ops_layer[1] = OP_RELU;
        //scale_layer[1] = 256; // 1.0 in fixed point
        $display("Bias: %0d", bias_layer[1]);
        $display("Activation: ReLU");
        
        // Configure Layer 3: Output Layer with Sigmoid
        $display("\n--- Layer 3 Configuration ---");
        $display("Type: Output Layer (MatMul + Bias + Sigmoid)");
        //layer3_weights[0] = 1; layer3_weights[1] = 0; layer3_weights[2] = 1;
        //layer3_weights[3] = 0; layer3_weights[4] = 1; layer3_weights[5] = 0;
        //layer3_weights[6] = 1; layer3_weights[7] = 0; layer3_weights[8] = 1;
        
        $display("Weights:");
        $display("[%0d %0d %0d]", layer3_weights[0], layer3_weights[1], layer3_weights[2]);
        $display("[%0d %0d %0d]", layer3_weights[3], layer3_weights[4], layer3_weights[5]);
        $display("[%0d %0d %0d]", layer3_weights[6], layer3_weights[7], layer3_weights[8]);
        
        //bias_layer[2] = 0;
        //vpu_ops_layer[2] = OP_RELU;
        //scale_layer[2] = 256; // 1.0 in fixed point
        $display("Bias: %0d", bias_layer[2]);
        $display("Activation: ReLU");
        
        // Reset
        $display("\n--- Starting Inference ---");
        #(CLK_PERIOD*2);
        rst = 0;
        #(CLK_PERIOD);
        
        // Start inference
        start = 1;
        #(CLK_PERIOD);
        start = 0;
        
        // Monitor progress
        $display("\nProcessing layers...");
        
        // Wait for completion
        wait(done);
        #(CLK_PERIOD*5);
        
        // Display final output
        $display("\n=========================================================");
        $display("INFERENCE COMPLETE");
        $display("=========================================================");
        $display("\nFinal Output (3x3):");
        $display("[%0d %0d %0d]", output_data[0], output_data[1], output_data[2]);
        $display("[%0d %0d %0d]", output_data[3], output_data[4], output_data[5]);
        $display("[%0d %0d %0d]", output_data[6], output_data[7], output_data[8]);
        
        $display("\n--- Summary ---");
        $display("Input → Layer1(ReLU) → Layer2(ReLU) → Layer3(Sigmoid) → Output");
        $display("Total Layers Processed: %0d", NUM_LAYERS);
        $display("=========================================================\n");
        
        #(CLK_PERIOD*10);
        $finish;
    end
    
    // Layer transition monitor
    reg [7:0] prev_layer;
    initial prev_layer = 0;
    
    always @(posedge clk) begin
        if (current_layer != prev_layer && !rst) begin
            $display("[Time %0t] Processing Layer %0d", $time, current_layer);
            prev_layer = current_layer;
        end
    end
    
    // State monitor (optional, for debugging)
    reg [3:0] prev_state;
    initial prev_state = 0;
    
    always @(posedge clk) begin
        if (state != prev_state && !rst) begin
            case (state)
                0: $display("  State: IDLE");
                1: $display("  State: LOAD_WEIGHTS");
                2: $display("  State: START_MATMUL");
                3: $display("  State: WAIT_MATMUL");
                4: $display("  State: APPLY_BIAS");
                5: $display("  State: APPLY_ACTIVATION");
                6: $display("  State: STORE_RESULT");
                7: $display("  State: NEXT_LAYER");
                8: $display("  State: COMPLETE");
            endcase
            prev_state = state;
        end
    end

endmodule
