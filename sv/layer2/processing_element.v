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
