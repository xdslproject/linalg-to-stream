#map = affine_map<(d0, d1) -> (d0 + d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
#map2 = affine_map<(d0, d1) -> (d0)>
"builtin.module"() ({
  "func.func"() <{function_type = (memref<16xf32>, memref<3xf32>, memref<14xf32>) -> (), sym_name = "conv1d_no_symbols"}> ({
  ^bb0(%arg0: memref<16xf32>, %arg1: memref<3xf32>, %arg2: memref<14xf32>):
    "linalg.generic"(%arg0, %arg1, %arg2) <{indexing_maps = [#map, #map1, #map2], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
      %0 = "arith.mulf"(%arg3, %arg4) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %1 = "arith.addf"(%arg5, %0) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%1) : (f32) -> ()
    }) : (memref<16xf32>, memref<3xf32>, memref<14xf32>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

