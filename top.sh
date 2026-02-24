#!/bin/bash -f

TOP_DIR=${PWD}

set size = 5

# generate new matrices
	rm -rf ./memories/*.hex
	python ./model/genmat.py --n ${size} --seed 1 --out $TOP_DIR/memories/inputs.hex --min 0 --max 20
	python ./model/genmat.py --n ${size} --seed 1 --out $TOP_DIR/memories/w0.hex --min 0 --max 2
	python ./model/genmat.py --n ${size} --seed 1 --out $TOP_DIR/memories/w1.hex --min 0 --max 2
	python ./model/genmat.py --n ${size} --seed 1 --out $TOP_DIR/memories/w2.hex --min 0 --max 2

#echo "\n\n\n Running Python Model, cmd : python ./model/ref_model.py ... \n\n\n"
	rm -rf ./memories/ref_output.hex ./memories/sim_output.hex

echo "\n\n\n Running Python Model, cmd : python ./model/ref_model.py ... \n\n\n"
	python ./model/refmodel.py --debug 1 --cols ${size} --rows ${size} --layers 3 --mempath $PWD/memories

echo "\n\n\n Running RTL Simulation, cmd : ./runme.sh ... \n\n\n"
	./runme.sh

echo "\n\n\n Comparing Python model to RTL, cmd : python ./model/compare.py ... \n\n\n"
	python ./model/compare.py --n ${size} --ref ./memories/ref_output.hex --sim ./memories/sim_output.hex


