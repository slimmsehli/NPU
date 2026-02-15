#!/bin/bash -f


rm -rf obj_dir

echo "\n\n\n Compiling ... \n\n\n"
verilator --binary -j 0 --trace -Wall \
	processing_element.v  systolic_array.v  systolic_controller.v  vector_processing_unit.v top_vpu.v \
	--top-module top \
	-Wno-UNDRIVEN -Wno-UNUSEDSIGNAL -Wno-WIDTHEXPAND -Wno-IMPLICIT -Wno-PINCONNECTEMPTY -Wno-DECLFILENAME \
	-Wno-BLKSEQ -Wno-IGNOREDRETURN -Wno-GENUNNAMED -Wno-UNUSEDPARAM

echo "\n\n\n Simulation ... \n\n\n"
./obj_dir/Vtop

#gtkwave waves.vcd



