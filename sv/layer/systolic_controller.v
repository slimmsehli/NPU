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
