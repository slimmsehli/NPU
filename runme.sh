#!/bin/bash -f


verilator --binary -j 0 --trace -Wall axi_interface.sv post_processing_unit.sv global_controller.sv processing_element.sv systolic_array.sv npu_top.sv --top-module npu_top




