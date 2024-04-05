input mlir:
```
"builtin.module"() ({
  %0:3 = "test.op"() : () -> (memref<16x16xf32>, memref<16x16xf32>, memref<16x16xf32>)
  "linalg.generic"(%0#0, %0#1, %0#2) <{indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operandSegmentSizes = array<i32: 2, 1>}> ({
  ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
    %1 = "arith.mulf"(%arg0, %arg1) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
    %2 = "arith.addf"(%arg2, %1) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
    "linalg.yield"(%2) : (f32) -> ()
  }) : (memref<16x16xf32>, memref<16x16xf32>, memref<16x16xf32>) -> ()
}) : () -> ()
```


input workload:
```
workload = {
    0: {
        "operator_type": "default",
        "equation": "O[d0][d1] += I[d0][d2] * W[d2][d1]",
        "dimension_relations": [],
        "loop_dim_size": {"D0": 16, "D1": 16, "D2": 16},
        "operand_precision": {"O": 32, "O_final": 32, "W": 32, "I": 32},
        "operand_source": {"W": [], "I": []},
        "constant_operands": ["I", "W"],
        "padding": {"D0": (0, 0), "D2": (0, 0)},
    }
}
```


```
========================================================================
Temporal Loops                   O            W            I            
========================================================================
for D0 in [0, 2):                l1           l1           l1           
------------------------------------------------------------------------
  for D1 in [0, 2):              l1           l1           l1           
------------------------------------------------------------------------
    for D2 in [0, 2):            rf_32b_O     l1           l1           
------------------------------------------------------------------------
========================================================================
Spatial Loops                                                           
========================================================================
      parfor D0 in [0, 8):                                              
------------------------------------------------------------------------
      parfor D1 in [0, 8):                                              
------------------------------------------------------------------------
      parfor D2 in [0, 8):                                              
------------------------------------------------------------------------
```
You will have scf for loops with a smaller linalg operation in side of it.

The smaller linalg inside of it can just be dispatched to the accelerator.

You start with linalg generic, then you generate

scf for nested for loops

inside these loops,

- you have tile selection and preparation (move to correct place in memory)

- linalg operation on a single tile (this is the part fed to an accelerator at a lower stage)



accelerator is connected to L1, so accelerator can handle any loops that only access L1

there is a level we control, and level accelerator controls they meet at level L1.

As soon as a loop level contains even one col entry with a higher level memory than L1, we need to care about that loop and implement the tiling.

We control memory above L1, and  only loading memory onto L1 before accelerator starts running

We put initial stuff on L1, but then acclerator read and writes on this level and below

For now, assume optimal solution of accelerator level unrolling coincides with accelerator's possiblities

As loop pseudocode, this output sort of looks like:

```
d0_bk_sz = 8
d1_bk_sz = 8
d2_bk_sz = 8

for(d0_1 = 0; d0_1 < 2; d_01 ++){
	for(d1_1 = 0; d1_1 < 2; d1_1 ++){
		for(d2_1 = 0; d2_1 < 2; d2_1++){
		
			// the rest of these loops are implicitly unrolled by the accelerator??
			for(d0_2 = 0; d0_2 < 8; d0_2 ++){				
				for(d1_2 = 0; d1_2 < 8; d1_2 ++){					
					for(d2_2 = 0; d2_2 < 8; d2_2++){
						d0 = d0_1 * d0_bk_sz + d0_2;
						d1 = d1_1 * d1_bk_sz + d1_2
						d2 = d2_1 * d2_bk_sz + d2_2
						O[d0][d1] = I[d0][d2] * W[d2][d1]
     					
					}
				}
			}
		} 
	}  
}

affine_map<(d0_1, d0_2, d1_1, d1_2, d2_1, d2_2) -> (d0, d2)>, Input
affine_map<(d0_1, d0_2, d1_1, d1_2, d2_1, d2_2) -> (d2, d1)>, Width
affine_map<(d0_1, d0_2, d1_1, d1_2, d2_1, d2_2) -> (d0, d1)>, Output

affine_map<(d0_1, d0_2, d1_1, d1_2, d2_1, d2_2) -> (d0_1, d0_2, d2_1, d2_2)>, Input
affine_map<(d0_1, d0_2, d1_1, d1_2, d2_1, d2_2) -> (d2, d1)>, Width
affine_map<(d0_1, d0_2, d1_1, d1_2, d2_1, d2_2) -> (d0, d1)>, Output
 
```





what is the desired output MLIR?

```
"builtin.module"() ({
  %0:3 = "test.op"() : () -> (memref<16x16xf32>, memref<16x16xf32>, memref<16x16xf32>)
  "linalg.generic"(%0#0, %0#1, %0#2) <{indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, 
                                                        affine_map<(d0, d1, d2) -> (d2, d1)>, 
                                                        affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operandSegmentSizes = array<i32: 2, 1>}> ({
  ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
    %1 = "arith.mulf"(%arg0, %arg1) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
    %2 = "arith.addf"(%arg2, %1) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
    "linalg.yield"(%2) : (f32) -> ()
  }) : (memref<16x16xf32>, memref<16x16xf32>, memref<16x16xf32>) -> ()
}) : () -> ()

"builtin.module"() ({
  %0:3 = "test.op"() : () -> (memref<16x16xf32>, memref<16x16xf32>, memref<16x16xf32>)
  "linalg.generic"(%0#0, %0#1, %0#2) <{indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operandSegmentSizes = array<i32: 2, 1>}> ({
  ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
    %1 = "arith.mulf"(%arg0, %arg1) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
    %2 = "arith.addf"(%arg2, %1) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
    "linalg.yield"(%2) : (f32) -> ()
  }) : (memref<16x16xf32>, memref<16x16xf32>, memref<16x16xf32>) -> ()
}) : () -> ()
```

## running gemm example notes

```
(.env) emily@ProfessorPlum:~/zigzag$ python main_gemm.py --model zigzag.inputs.examples.workload.gemm_layer --mapping zigzag.inputs.examples.mapping.gemm --accelerator zigzag.inputs.examples.hardware.Gemm
2024-04-05 09:41:05,160 - INFO - Created workload graph with 1 nodes and 0 edges.
2024-04-05 09:41:05,160 - INFO - Parsed accelerator with cores [1].
2024-04-05 09:41:05,160 - INFO - Processing layer 0...
2024-04-05 09:41:05,160 - INFO - Launching spatial mapping 1/1: {'D1': ('M', 8), 'D2': ('N', 8), 'D3': ('K', 8)}.
2024-04-05 09:41:05,167 - INFO - Saved CostModelEvaluation(layer=LayerNode_0, core=1) with energy 6.140e+07 and latency 6.074e+05 to outputs/Gemm-gemm_layer-LayerNode_0_complete.json
=============================================================================
Temporal Loops                        O            B            A            
=============================================================================
for M in [0, 8):                      l3           l3           l3           
-----------------------------------------------------------------------------
  for N in [0, 8):                    l3           l3           l3           
-----------------------------------------------------------------------------
    for K in [0, 8):                  l1           l3           l3           
-----------------------------------------------------------------------------
      for M in [0, 8):                l1           l1           l1           
-----------------------------------------------------------------------------
        for N in [0, 8):              l1           l1           l1           
-----------------------------------------------------------------------------
          for K in [0, 8):            rf_32b_O     l1           l1           
-----------------------------------------------------------------------------
=============================================================================
Spatial Loops                                                                
=============================================================================
            parfor M in [0, 8):                                              
-----------------------------------------------------------------------------
            parfor N in [0, 8):                                              
-----------------------------------------------------------------------------
            parfor K in [0, 8):                                              
-----------------------------------------------------------------------------
Stall and slack per port of each memory instance:
  rf_32b_O: {'r_port_1': 0, 'r_port_2': 0, 'w_port_1': 0, 'w_port_2': 0}
  l1: {'rw_port_1': 345036}
  l3: {'rw_port_1': 0}
Latency: 6.074e+05

```

