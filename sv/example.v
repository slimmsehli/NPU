// NPU Configuration Examples
// Shows how to configure the NPU for different network architectures

// ============================================================================
// EXAMPLE 1: Simple Binary Classifier
// ============================================================================
// Architecture: Input(3×3) → Hidden(ReLU) → Hidden(ReLU) → Output(Sigmoid)
// Use case: Binary classification (e.g., cat vs dog)

module example_classifier;
    parameter NUM_LAYERS = 3;
    
    // Layer 1: Input → First Hidden Layer
    // 3×3 input → 3×3 hidden with ReLU
    assign bias_layer[0] = 5;
    assign vpu_ops_layer[0] = 3'd1;      // ReLU activation
    assign scale_layer[0] = 256;          // No scaling (1.0)
    
    // Layer 2: Hidden → Second Hidden Layer  
    // 3×3 hidden → 3×3 hidden with ReLU
    assign bias_layer[1] = 3;
    assign vpu_ops_layer[1] = 3'd1;      // ReLU activation
    assign scale_layer[1] = 256;          // No scaling (1.0)
    
    // Layer 3: Hidden → Output Layer
    // 3×3 hidden → 3×3 output with Sigmoid
    assign bias_layer[2] = 0;
    assign vpu_ops_layer[2] = 3'd6;      // Sigmoid activation
    assign scale_layer[2] = 256;          // No scaling (1.0)
    
    // Output interpretation:
    // output_data[4] (center element) could be used as binary decision
    // > threshold → class 1, < threshold → class 0
endmodule


// ============================================================================
// EXAMPLE 2: Feature Extractor with Pooling
// ============================================================================
// Architecture: Input(3×3) → Conv-like → Max Pool → Output
// Use case: Extract high-level features from input

module example_feature_extractor;
    parameter NUM_LAYERS = 2;
    
    // Layer 1: Input → Feature Layer with ReLU
    // Extract features from input
    assign bias_layer[0] = 0;
    assign vpu_ops_layer[0] = 3'd1;      // ReLU activation
    assign scale_layer[0] = 256;
    
    // Layer 2: Features → Pooled Features
    // Downsample to most important features
    assign bias_layer[1] = 0;
    assign vpu_ops_layer[1] = 3'd4;      // Max Pooling
    assign scale_layer[1] = 256;
    
    // Output: 4 most prominent features (after 2×2 pooling)
    // output_data[0], output_data[1], output_data[3], output_data[4]
endmodule


// ============================================================================
// EXAMPLE 3: Normalized Deep Network
// ============================================================================
// Architecture: Input → Hidden(Norm) → Hidden(ReLU) → Output(Sigmoid)
// Use case: Network with batch normalization

module example_normalized_network;
    parameter NUM_LAYERS = 3;
    
    // Layer 1: Input → Normalized Hidden Layer
    // Apply normalization (scaling) followed by ReLU
    assign bias_layer[0] = 0;
    assign vpu_ops_layer[0] = 3'd3;      // Scale operation
    assign scale_layer[0] = 128;          // Scale by 0.5
    
    // Layer 2: Hidden → Hidden with ReLU
    assign bias_layer[1] = 10;
    assign vpu_ops_layer[1] = 3'd1;      // ReLU activation
    assign scale_layer[1] = 256;
    
    // Layer 3: Hidden → Output with Sigmoid
    assign bias_layer[2] = 0;
    assign vpu_ops_layer[2] = 3'd6;      // Sigmoid activation
    assign scale_layer[2] = 256;
endmodule


// ============================================================================
// EXAMPLE 4: Regression Network (No Final Activation)
// ============================================================================
// Architecture: Input → Hidden(ReLU) → Hidden(ReLU) → Output(Linear)
// Use case: Predict continuous values

module example_regression_network;
    parameter NUM_LAYERS = 3;
    
    // Layer 1: Input → First Hidden
    assign bias_layer[0] = 5;
    assign vpu_ops_layer[0] = 3'd1;      // ReLU
    assign scale_layer[0] = 256;
    
    // Layer 2: Hidden → Second Hidden
    assign bias_layer[1] = 3;
    assign vpu_ops_layer[1] = 3'd1;      // ReLU
    assign scale_layer[1] = 256;
    
    // Layer 3: Hidden → Output (Linear)
    assign bias_layer[2] = 0;
    assign vpu_ops_layer[2] = 3'd0;      // Passthrough (no activation)
    assign scale_layer[2] = 256;
    
    // Output: Continuous values for regression
endmodule


// ============================================================================
// EXAMPLE 5: Multi-Stage Processing
// ============================================================================
// Architecture: Input → Process1 → Process2 → Process3 → Output
// Each stage does different operation

module example_multi_stage;
    parameter NUM_LAYERS = 4;
    
    // Stage 1: Apply transformation with bias
    assign bias_layer[0] = 10;
    assign vpu_ops_layer[0] = 3'd2;      // Bias add only
    assign scale_layer[0] = 256;
    
    // Stage 2: Activate
    assign bias_layer[1] = 0;
    assign vpu_ops_layer[1] = 3'd1;      // ReLU
    assign scale_layer[1] = 256;
    
    // Stage 3: Scale down
    assign bias_layer[2] = 0;
    assign vpu_ops_layer[2] = 3'd3;      // Scale
    assign scale_layer[2] = 128;          // 0.5x
    
    // Stage 4: Final activation
    assign bias_layer[3] = 0;
    assign vpu_ops_layer[3] = 3'd7;      // Tanh
    assign scale_layer[3] = 256;
endmodule


// ============================================================================
// EXAMPLE 6: Pattern Detector
// ============================================================================
// Architecture: Input → Edge Detect → Feature → Decision
// Use case: Detect specific patterns in input

module example_pattern_detector;
    parameter NUM_LAYERS = 3;
    
    // Layer 1: Edge detection (using specific weight pattern)
    // Weights would be configured as edge detection kernel
    assign bias_layer[0] = 0;
    assign vpu_ops_layer[0] = 3'd1;      // ReLU (keep positive edges)
    assign scale_layer[0] = 256;
    
    // Layer 2: Feature combination
    assign bias_layer[1] = 5;
    assign vpu_ops_layer[1] = 3'd1;      // ReLU
    assign scale_layer[1] = 256;
    
    // Layer 3: Pattern decision
    assign bias_layer[2] = -10;           // Negative bias for threshold
    assign vpu_ops_layer[2] = 3'd6;      // Sigmoid
    assign scale_layer[2] = 256;
    
    // Output: High values where pattern is detected
endmodule


// ============================================================================
// CONFIGURATION GUIDELINES
// ============================================================================

/*
BIAS SELECTION:
- Hidden layers: Typically 0-10
- Output layers: 0 or small value
- Negative bias: Acts as threshold

ACTIVATION FUNCTIONS:
- ReLU (3'd1): Most common for hidden layers, prevents vanishing gradient
- Sigmoid (3'd6): Binary classification outputs, values 0-1
- Tanh (3'd7): Hidden layers when you want -1 to +1 range
- Passthrough (3'd0): Linear output or when chaining operations

SCALE FACTORS (with FRAC_BITS=8):
- 256 = 1.0 (no scaling)
- 128 = 0.5 (halve values)
- 512 = 2.0 (double values)
- Use for normalization or preventing overflow

POOLING:
- Max Pool (3'd4): After feature extraction, keeps strongest features
- Avg Pool (3'd5): Smoother downsampling
- Reduces output size to 4 values

TYPICAL PATTERNS:
1. Classification: MatMul → Bias → ReLU → ... → MatMul → Bias → Sigmoid
2. Regression: MatMul → Bias → ReLU → ... → MatMul → Bias → Passthrough
3. Feature Extract: MatMul → Bias → ReLU → MaxPool
4. Normalized: MatMul → Scale → Bias → ReLU
*/


// ============================================================================
// WEIGHT CONFIGURATION EXAMPLES
// ============================================================================

module weight_examples;
    
    // Identity-like weights (preserve input structure)
    reg [7:0] identity_weights [0:8];
    initial begin
        identity_weights[0] = 1; identity_weights[1] = 0; identity_weights[2] = 0;
        identity_weights[3] = 0; identity_weights[4] = 1; identity_weights[5] = 0;
        identity_weights[6] = 0; identity_weights[7] = 0; identity_weights[8] = 1;
    end
    
    // Averaging weights (smooth input)
    reg [7:0] averaging_weights [0:8];
    initial begin
        averaging_weights[0] = 1; averaging_weights[1] = 1; averaging_weights[2] = 1;
        averaging_weights[3] = 1; averaging_weights[4] = 1; averaging_weights[5] = 1;
        averaging_weights[6] = 1; averaging_weights[7] = 1; averaging_weights[8] = 1;
    end
    
    // Edge detection weights (vertical edges)
    reg [7:0] edge_weights [0:8];
    initial begin
        edge_weights[0] = 8'd255; edge_weights[1] = 0; edge_weights[2] = 1;  // -1, 0, 1
        edge_weights[3] = 8'd255; edge_weights[4] = 0; edge_weights[5] = 1;
        edge_weights[6] = 8'd255; edge_weights[7] = 0; edge_weights[8] = 1;
    end
    
    // Center-focused weights (emphasize center pixel)
    reg [7:0] center_weights [0:8];
    initial begin
        center_weights[0] = 0; center_weights[1] = 1; center_weights[2] = 0;
        center_weights[3] = 1; center_weights[4] = 4; center_weights[5] = 1;
        center_weights[6] = 0; center_weights[7] = 1; center_weights[8] = 0;
    end
    
endmodule


// ============================================================================
// COMPLETE EXAMPLE: Image Classifier
// ============================================================================

module complete_classifier_example;
    
    /*
    Task: Classify 3×3 binary images (e.g., X vs O patterns)
    
    Network Architecture:
    - Input: 3×3 binary image (0s and 1s)
    - Layer 1: Feature extraction with ReLU
    - Layer 2: Feature combination with ReLU
    - Layer 3: Classification with Sigmoid
    - Output: Classification score (0-1)
    
    Training: Weights would be trained offline, then loaded here
    */
    
    parameter NUM_LAYERS = 3;
    
    // Input example: X pattern
    // [1 0 1]
    // [0 1 0]
    // [1 0 1]
    reg [7:0] input_x_pattern [0:8];
    initial begin
        input_x_pattern[0] = 1; input_x_pattern[1] = 0; input_x_pattern[2] = 1;
        input_x_pattern[3] = 0; input_x_pattern[4] = 1; input_x_pattern[5] = 0;
        input_x_pattern[6] = 1; input_x_pattern[7] = 0; input_x_pattern[8] = 1;
    end
    
    // Layer 1: Detect diagonal features
    reg [7:0] layer1_weights [0:8];
    assign bias_layer[0] = 2;
    assign vpu_ops_layer[0] = 3'd1;  // ReLU
    
    // Layer 2: Combine features
    reg [7:0] layer2_weights [0:8];
    assign bias_layer[1] = 5;
    assign vpu_ops_layer[1] = 3'd1;  // ReLU
    
    // Layer 3: Final classification
    reg [7:0] layer3_weights [0:8];
    assign bias_layer[2] = 0;
    assign vpu_ops_layer[2] = 3'd6;  // Sigmoid
    
    /*
    Usage:
    1. Load input_x_pattern into NPU
    2. Start inference
    3. Wait for done
    4. Read output_data[4] (center value) as classification score
    5. If score > 0.5 → "X" detected, else "O" detected
    */
    
endmodule
