import zigzag.api
from zigzag.visualization.results.print_mapping import print_mapping

# what we wish would work
mapping= "inputs.mapping.snax_gemm"
accelerator = "inputs.hardware.snax_gemm"
workload = "workload"

# gemm example from gemm-example branch
# mapping= "inputs.mapping.gemm"
# accelerator = "inputs.hardware.Gemm"
# workload = "inputs.workload.gemm_layer"

# what might work, but workload comes from stream so maybe other problems?
# mapping= "inputs.mapping.snax_gemm"
# accelerator = "inputs.hardware.snax_gemm"
# workload = "inputs.workload.testing_workload_for_1_core"

# energy, latency, cme = get_hardware_performance_zigzag(
#     workload,
#     accelerator,
#     mapping,
#     opt="latency",
#     dump_filename_pattern="outputs/{datetime}.json",
#     pickle_filename="outputs/list_of_cmes.pickle"
# )

answers = zigzag.api.get_hardware_performance_zigzag(workload, accelerator, mapping, "latency", "outputs/my-output.json","outputs/list_of_cmes.pickle")


# print(f'answers is a {type(answers)} with length {len(answers)}')
# print(f'answers[0] is a {type(answers[0])}')# with length {len(answers[0])}')
# print(f'answers[1] is a {type(answers[1])}')# with length {len(answers[1])}')
# print(f'answers[2] is a {type(answers[2])} with length {len(answers[2])}')
# print(f'answers[2][0] is a {type(answers[2][0])} with length {len(answers[2][0])}')#and answers[2][1] is a {type(answers[2][1])}')
# print(f'answers[2][0][0] is a {type(answers[2][0][0])}')
# print(f'answers[2][0][1] is a {type(answers[2][0][1])} with length {len(answers[2][0][1])}')

cme = answers[2][0][0]
print(cme.temporal_mapping)
print(cme.spatial_mapping)