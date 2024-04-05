"builtin.module"() ({
  %0, %1, %2, %3, %4 = "test.op"() : () -> (memref<16x20xi8>, memref<20x24xi8>, i32, i32, memref<16x24xi32>)
  "linalg.generic"(%0, %1, %2, %3, %4) <{"indexing_maps" = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> (d0, d1)>], "iterator_types" = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], "operandSegmentSizes" = array<i32: 4, 1>}> ({
  ^0(%arg3 : i8, %arg4 : i8, %arg5 : i32, %arg6 : i32, %arg7 : i32):
    %5 = "arith.extsi"(%arg3) : (i8) -> i32
    %6 = "arith.subi"(%5, %arg5) : (i32, i32) -> i32
    %7 = "arith.extsi"(%arg4) : (i8) -> i32
    %8 = "arith.subi"(%7, %arg6) : (i32, i32) -> i32
    %9 = "arith.muli"(%6, %8) : (i32, i32) -> i32
    %10 = "arith.addi"(%arg7, %9) : (i32, i32) -> i32
    "linalg.yield"(%10) : (i32) -> ()
  }) : (memref<16x20xi8>, memref<20x24xi8>, i32, i32, memref<16x24xi32>) -> ()
}) : () -> ()
