from xdsl.dialects import builtin, memref
from xdsl.dialects.linalg import Generic
from xdsl.ir import MLContext
from xdsl.ir.affine import AffineDimExpr, AffineMap
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.dialects.builtin import (
    IntegerType
)

from util.kernel_type import KernelType


class LinalgToStreamTranslator(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, generic_op: Generic, rewriter: PatternRewriter):

        kernel_type = KernelType.get_kernel(generic_op)

        if not kernel_type:
            return
        
        # if kernel_type != KernelType.MUL:
        #     print(f"kernel type of {kernel_type} not yet supported")
        #     return

        # zigzag -> 3 operands: 2 inputs, 1 output
        assert len(generic_op.outputs) == 1
        generic_op.outputs[0]

        assert len(generic_op.inputs) == 2
        generic_op.inputs[0]
        generic_op.inputs[1]

        zigzag_description = dict()

        # for now, set operator to default type
        zigzag_description["operator_type"] = "default"

        # construct equation
        output_access = "O"
        for i in range(len(generic_op.indexing_maps.data[-1].data.results)):
            map = generic_op.indexing_maps.data[-1].data.results[i]
            assert isinstance(map, AffineDimExpr)
            output_access += f"[{str(map)}]"
       
        input_i_access = "I"
        for i in range(len(generic_op.indexing_maps.data[0].data.results)):
            map = generic_op.indexing_maps.data[0].data.results[i]
            assert isinstance(map, AffineDimExpr)
            input_i_access += f"[{str(map)}]"

        input_w_access = "W"
        for i in range(len(generic_op.indexing_maps.data[1].data.results)):
            map = generic_op.indexing_maps.data[1].data.results[i]
            assert isinstance(map, AffineDimExpr)
            input_w_access += f"[{str(map)}]"

        if kernel_type == KernelType.MUL:
            zigzag_description[
                "equation"
            ] = f"{output_access} = {input_i_access} * {input_w_access}"

        elif kernel_type in (KernelType.MAC, KernelType.QMAC):
            zigzag_description[
                "equation"
            ] = f"{output_access} += {input_i_access} * {input_w_access}"

        # extract dimension_relations (for now, leave empty)
        zigzag_description["dimension_relations"] = []

        # extract loop bounds
        results = []
        results.extend(generic_op.indexing_maps.data[0].data.results)
        results.extend(generic_op.indexing_maps.data[1].data.results)
        results.extend(generic_op.indexing_maps.data[2].data.results)

        combined_affine_map = AffineMap(3, 0, results)
        inverse_map = combined_affine_map.inverse_permutation()

        memref_shapes = [shape.data for op in generic_op.operands for shape in op.type.shape.data]
        iteration_bounds = inverse_map.eval(memref_shapes, [])

        zigzag_description["loop_dim_size"] = dict()

        for i, bound in enumerate(iteration_bounds):
            zigzag_description["loop_dim_size"][f"d{i}"] = bound

        # extract operand precision
        widths = []
        for op in generic_op.operands :
            assert isinstance(op.type, memref.MemRefType)           
            element_type = op.type.get_element_type()
            if(isinstance(element_type, IntegerType)):
                widths.append(element_type.width.data)
            else:
                widths.append(element_type.get_bitwidth)
        
        zigzag_description["operand_precision"] = dict()
        zigzag_description["operand_precision"]["O"] = widths[-1]
        zigzag_description["operand_precision"]["O_final"] = widths[-1]
        zigzag_description["operand_precision"]["W"] = widths[0]
        zigzag_description["operand_precision"]["I"] = widths[1]        
  
        # operand source (use default of no source for now)
        zigzag_description["operand_source"] = dict()
        zigzag_description["operand_source"]["W"] = []
        zigzag_description["operand_source"]["I"] = []

        # constant operands (use default of all constant operands for now)
        zigzag_description["constant_operands"] = dict()
        zigzag_description["constant_operands"] = ["I", "W"]

        # padding (use default of 0 padding for now)
        # affects last two indices of input I
        zigzag_description["padding"] = dict()
        zigzag_description["padding"][str(generic_op.indexing_maps.data[0].data.results[0]).upper()] = (0,0)
        zigzag_description["padding"][str(generic_op.indexing_maps.data[0].data.results[1]).upper()] = (0,0)
        print(f"workload = {zigzag_description}")
        print("")

        pass


class LinalgToStream(ModulePass):
    name = "linalg-to-stream"

    def apply(self, ctx: MLContext, module: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            LinalgToStreamTranslator(), apply_recursively=False
        ).rewrite_module(module)
