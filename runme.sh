#!/bin/bash -f


verilator --binary -j 0 --trace -Wall \
	axi_interface.sv post_processing_unit.sv global_controller.sv processing_element.sv systolic_array.sv npu_top.sv top.sv \
	--top-module top \
	-Wno-UNDRIVEN -Wno-UNUSEDSIGNAL -Wno-WIDTHEXPAND -Wno-IMPLICIT -Wno-PINCONNECTEMPTY -Wno-DECLFILENAME

./obj_dir/Vtop

gtkwave waves.vcd



