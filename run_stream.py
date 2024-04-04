import stream.api


accelerator = "inputs.hardware.snax_gemm"
workload = "workload"
mapping= "inputs.mapping.snax_gemm"

stream.api.get_hardware_performance_stream(accelerator, workload, mapping, 1, [],"blah")
