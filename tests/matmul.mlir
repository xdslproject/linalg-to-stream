"builtin.module"() ({
  %0:3 = "test.op"() : () -> (memref<16x16xf32>, memref<16x16xf32>, memref<16x16xf32>)
  "linalg.generic"(%0#0, %0#1, %0#2) <{indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operandSegmentSizes = array<i32: 2, 1>}> ({
  ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
    %1 = "arith.mulf"(%arg0, %arg1) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
    %2 = "arith.addf"(%arg2, %1) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
    "linalg.yield"(%2) : (f32) -> ()
  }) : (memref<16x16xf32>, memref<16x16xf32>, memref<16x16xf32>) -> ()
}) : () -> ()
