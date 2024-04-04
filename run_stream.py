import stream.api

# accelerator = "inputs.testing.hardware.dual_testing_core_offchip"
# workload = "inputs.testing.workload.testing_workload_for_2_cores"
# mapping = "inputs.testing.mapping.testing_mapping"
# stream.api.get_hardware_performance_stream("inputs.hardware.snax_gemm", "workload","inputs.hardware.mapping_c_k",0,0,0) 

# [("OY", "all")]

# stream.api.get_hardware_performance_stream("inputs.hardware.snax_gemm", "workload","inputs.hardware.mapping_c_k",1,[],"blah")

# stream.api.get_hardware_performance_stream("inputs.hardware.snax_gemm", "workload","inputs.hardware.mapping_c_k",1,[("OY", "all")],"blah")

# linalg-input-output/inputs/examples/hardware/snax_gemm.py

# stream.api.get_hardware_performance_stream("inputs/examples/hardware/snax_gemm", "workload","inputs.hardware.mapping_c_k",1,[("OY", "all")],"blah")

# def get_hardware_performance_stream(
#     hardware, workload, mapping, CN_define_mode, hint_loops, node_hw_cost_pkl_name

accelerator = "inputs.hardware.snax_gemm"
workload = "workload"
mapping= "inputs.mapping.snax_gemm"

stream.api.get_hardware_performance_stream(accelerator, workload,mapping,1,[],"blah")