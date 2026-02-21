#!/bin/bash -f

TOP_DIR=${PWD}

echo "\n\n\n Cleaning ... \n\n\n"
	rm -rf simdir *.log *.vcd *.hex

echo "\n\n\n Compiling ... \n\n\n"
verilator --binary -j 0 --trace -Wall \
	-F ./sv/layer2/filelist.f \
	--top top --Mdir simdir -o simv \
	-Wno-UNDRIVEN -Wno-UNUSEDSIGNAL -Wno-WIDTHEXPAND -Wno-IMPLICIT -Wno-PINCONNECTEMPTY -Wno-DECLFILENAME -Wno-BLKSEQ \
	-Wno-UNUSEDPARAM -Wno-WIDTHTRUNC -Wno-VARHIDDEN \
	|& tee ./simdir/compile.log 

echo "\n\n\n Simulation ... \n\n\n"
./simdir/simv +WEIGHT_PATH=$PWD/memories |& tee ./simdir/simulation.log 

echo "\n\n\n Openining Waves ... \n\n\n"
#gtkwave waves.vcd &

