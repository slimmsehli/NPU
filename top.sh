#!/bin/bash -f

TOP_DIR=${PWD}

#echo "\n\n\n Running Python Model, cmd : python ./model/ref_model.py ... \n\n\n"
	rm -rf ./memories/ref_output.hex ./memories/sim_output.hex

echo "\n\n\n Running Python Model, cmd : python ./model/ref_model.py ... \n\n\n"
	python ./model/refmodel.py --debug 1 --cols 3 --rows 3 --layers 3 --mempath $PWD/memories

echo "\n\n\n Running RTL Simulation, cmd : ./runme.sh ... \n\n\n"
	./runme.sh

echo "\n\n\n Comparing Python model to RTL, cmd : python ./model/compare.py ... \n\n\n"
	python ./model/compare.py --ref ./memories/ref_output.hex --sim ./memories/sim_output.hex


