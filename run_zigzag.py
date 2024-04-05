import zigzag.api
from zigzag.visualization.results.print_mapping import print_mapping

# takes in workload generated from a run of xdsl_opt_main.py
mapping= "inputs.mapping.snax_gemm"
accelerator = "inputs.hardware.snax_gemm"
workload = "workload"

# gemm example from zizag's gemm-example branch
# mapping= "inputs.mapping.gemm"
# accelerator = "inputs.hardware.Gemm"
# workload = "inputs.workload.gemm_layer"

answers = zigzag.api.get_hardware_performance_zigzag(workload, accelerator, mapping, "latency", "outputs/my-output.json","outputs/list_of_cmes.pickle")

cme = answers[2][0][0]
print(cme.temporal_mapping)
print(cme.spatial_mapping)

print_mapping(cme)
