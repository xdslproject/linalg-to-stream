builtin.module {
  %0, %1, %2 = "test.op"() : () -> (memref<16x16xf32>, memref<16x16xf32>, memref<16x16xf32>)
  linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%0, %1 : memref<16x16xf32>, memref<16x16xf32>) outs(%2 : memref<16x16xf32>) attrs =  {"zigzag_stream_id" = #builtin.int<0>} {
  ^0(%arg0 : f32, %arg1 : f32, %arg2 : f32):
    %3 = arith.mulf %arg0, %arg1 : f32
    %4 = arith.addf %arg2, %3 : f32
    linalg.yield %4 : f32
  }
}

