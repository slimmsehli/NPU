


module top;

reg clk, reset;

npu_axi_stream_if slave_if();
npu_axi_stream_if master_if();

npu_top #(
    .ARRAY_SIZE(4),
    .DATA_WIDTH(8)
)
npu_top_inst (
	.clk(clk),
	.rst_n(reset),
	.s_axis(slave_if), // AXI4-Stream Slave (for Weights/Inputs)
	.m_axis(master_if) // AXI4-Stream Master (for Results)
);


//###############################################
//  Clock and Reset Block
initial begin
	reset = 0;
	clk = 0;
	repeat (2) @(posedge clk);
	reset = 1;
	repeat (2) @(posedge clk);
	reset = 0;
end
always #1 clk<=~clk;


//###############################################
// signal dumping
initial begin
		$dumpfile("waves.vcd");
        $dumpvars(0, npu_top_inst);
    	#10;
    	$finish;
    end

endmodule
