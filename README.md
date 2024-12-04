# Linalg to Stream

## End-Goal Linalg-to-Stream Tool Responsibilities

1. take in as input MLIR with linalg operations
2. for each linalg generic operation, 
   1. generate an equivalent ZigZag workload object
   2. feed this workload + a hardware description to zigzag
   3. parse the ZigZag output, including converting the prescribed loop bounds into tile sizes
   4. annotate the linalg operation with the ZigZag-prescribed tile size

3. output annotated MLIR, ***including an accompanying transform dialect script encoding the tiling transformation of each linalg operation***

![](pics/linalg-to-stream-tool.png)

## Current Functionality

Annotates a linalg matmul with a unique ID, and outputs corresponding ZigZag workload to a file.

Subtasks to expand functionality:

- generate workload as yaml file
- use Jonas's code to feed to zigzag and extract?
- use Jonas's code to create transform dialect script?
- how to make this a through-line that is easily extendable?

```
sh run.sh tests/matmul.mlir inputs/hardware/snax_gemm.py inputs/mapping/snax_gemm.py tests/out/myWorkload.yaml
```

What does the yaml file that I want to generate look like?

### Emily notes

https://github.com/EmilySillars/Quidditch-zigzag/blob/tiling/runtime/tests/tiledMatmul12/README.md

```
- id: 0 
  name: matmul_104_104  # name can be used to specify mapping
  operator_type: MatMul  # operator_type can be used to specify mapping
  equation: O[a][b]+=I[a][c]*W[c][b]
  dimension_relations: []
  loop_dims: [A,B,C]
  loop_sizes: [104, 104, 104]
  operand_precision:
    W: 8
    I: 8
    O: 32
    O_final: 32
  operand_source:
    I: 0
    W: 0
```



```
- id: 30 # fc
  operator_type: Conv
  equation: O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][iy][ix]
  dimension_relations: [ix=1*ox+1*fx, iy=1*oy+1*fy]
  loop_dims: [B, K, C, OY, OX, FY, FX]
  loop_sizes: [1, 1000, 512, 1, 1, 1, 1]
  operand_precision:
    W: 8
    I: 8
    O: 16
    O_final: 8
  operand_source:
    I: 29
    W: 30
```



## Set up

Install dependencies

```
pip install -r requirements.txt
```

## Running the tool

#### Run an Example

```
sh run.sh tests/matmul.mlir inputs/hardware/snax_gemm.py inputs/mapping/snax_gemm.py ./myWorkload.yaml
```

To view zigzag workload, do

```
cat ./myWorkload.yaml
```

#### Run a Regression Test

```
sh run_test.sh matmul snax_gemm.py snax_gemm.py
```

To view zigzag workload, do

```
cat tests/workloads/matmul.workload.out.yaml
```



...

## old documentation below - less relevant, but preserving here for reference

## running the tool

```
python xdsl_opt_main.py tests/matmul.mlir -p linalg-to-stream
```

### limitations

- Currently tool can only take in a single linalg generic operation
- The linalg generic operation must be a matrix multiply

### future work

- handle multiple linalg generic operations, assigning a unique id to each, which is then added as as attribute to the mlir operation

- figure out dependencies between linalg generic operations, and record this relationship in the workload objects using the unique ids

- handle case where linalg generic has more than three operands (we are assuming the first two operands are inputs, and the last operand is an output) Quantized operations have more than two inputs, and we would like to support these.

  Example of a quantized operation we want to support:

  ```
  func.func @simple_matmul(%arg0: memref<16x16xi8>, %arg1: memref<16x16xi8, strided<[1, 16], offset:0>>, %arg2: memref<16x16xi32>) {
  %c0_i32 = arith.constant 0 : i32
  linalg.quantized_matmul ins(%arg0, %arg1, %c0_i32, %c0_i32 : memref<16x16xi8>, memref<16x16xi8, strided<[1, 16], offset:0>>, i32, i32) outs(%arg2 : memref<16x16xi32>)
  return
  }
  ```

  gobolt.org MLIR opt (trunk) `--linalg-generalize-named-ops --mlir-print-local-scope --mlir-print-op-generic`

  ```
  "builtin.module"() ({
    "func.func"() <{function_type = (memref<16x16xi8>, memref<16x16xi8, strided<[1, 16]>>, memref<16x16xi32>) -> (), sym_name = "simple_matmul"}> ({
    ^bb0(%arg0: memref<16x16xi8>, %arg1: memref<16x16xi8, strided<[1, 16]>>, %arg2: memref<16x16xi32>):
      %0 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      "linalg.generic"(%arg0, %arg1, %0, %0, %arg2) <{indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operandSegmentSizes = array<i32: 4, 1>}> ({
      ^bb0(%arg3: i8, %arg4: i8, %arg5: i32, %arg6: i32, %arg7: i32):
        %1 = "arith.extsi"(%arg3) : (i8) -> i32
        %2 = "arith.subi"(%1, %arg5) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %3 = "arith.extsi"(%arg4) : (i8) -> i32
        %4 = "arith.subi"(%3, %arg6) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %5 = "arith.muli"(%2, %4) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %6 = "arith.addi"(%arg7, %5) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        "linalg.yield"(%6) : (i32) -> ()
      }) : (memref<16x16xi8>, memref<16x16xi8, strided<[1, 16]>>, i32, i32, memref<16x16xi32>) -> ()
      "func.return"() : () -> ()
    }) : () -> ()
  }) : () -> ()
  ```

Note: None of the MLIR matrix operations take padding as input, so we know all the workloads produced from linalg should have 0 padding.

Tests To Make:

- vector x vector ; elementwise multiplication
- conv2D
- Conv 1D
- Depthwise Conv2D*
- Pointwise Conv2D
- Matrix-vector multi.
- matrix-martix multiply

### feed output of tool into zigzag

```
python run_zigzag.py 
```
### feed output of tool into stream (need to fix)

```
python run_stream.py
```
[Errors we're getting documented here](https://github.com/EmilySillars/stream-zigzag-input-output-linalg/tree/add-linalg-as-output-from-stream/linalg-input-output)
