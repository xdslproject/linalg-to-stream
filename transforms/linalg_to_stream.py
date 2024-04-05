from xdsl.dialects import builtin, memref
from xdsl.dialects.linalg import Generic
from xdsl.ir import MLContext
from xdsl.ir.affine import AffineBinaryOpExpr, AffineDimExpr, AffineMap, AffineExpr
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.dialects.builtin import IntegerType, IntAttr

from util.kernel_type import KernelType


class LinalgToStreamTranslator(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, generic_op: Generic, rewriter: PatternRewriter):
        # extract the kernel type from the generic op
        # as zigzag only works with MAC, MUL, ...
        kernel_type = KernelType.get_kernel(generic_op)

        if not kernel_type:
            return

        # zigzag -> 3 operands: 2 inputs, 1 output
        assert len(generic_op.outputs) == 1
        generic_op.outputs[0]

        assert len(generic_op.inputs) == 2
        generic_op.inputs[0]
        generic_op.inputs[1]

        zigzag_description = dict()
        dimension_relations = []

        # for now, set operator to default type
        zigzag_description["operator_type"] = "default"

        # construct equation
        output_access = "O"
        for i in range(len(generic_op.indexing_maps.data[0].data.results)):
            map = generic_op.indexing_maps.data[-1].data.results[i]
            assert isinstance(map, AffineDimExpr)
            output_access += f"[{str(map)}]"

        input_i_access = "I"
        for i in range(len(generic_op.indexing_maps.data[0].data.results)):
            map = generic_op.indexing_maps.data[0].data.results[i]
            if isinstance(map, AffineBinaryOpExpr):
                dimension_relations.append(
                    "ix=" + str(map.lhs) + str(map.kind.get_token()) + str(map.rhs)
                )
                input_i_access += f"[ix]"
            else:
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
        print(zigzag_description)

        # extract dimension_relations
        # for matmul, this is empty
        zigzag_description["dimension_relations"] = dimension_relations

        # extract loop bounds by evaluating the inverse affine map
        # with the memref shapes as input
        # results = []

        num_dims = max([dim.data.num_dims for dim in generic_op.indexing_maps.data])
        print(num_dims)
        # results.extend(generic_op.indexing_maps.data[0].data.results)
        # results.extend(generic_op.indexing_maps.data[1].data.results)
        # results.extend(generic_op.indexing_maps.data[2].data.results)
        # results.extend(num_dims)

        # combined_affine_map = AffineMap(num_dims, 0, results)
        combined_affine_map = AffineMap(
            num_dims, 0, (AffineExpr.dimension(0), AffineExpr.dimension(1))
        )
        inverse_map = combined_affine_map.inverse_permutation()
        print(combined_affine_map)
        print(inverse_map)

        memref_shapes = [
            shape.data for op in generic_op.operands for shape in op.type.shape.data
        ]
        iteration_bounds = inverse_map.eval(memref_shapes[:num_dims], [])
        print(iteration_bounds)
        zigzag_description["loop_dim_size"] = dict()

        for i, bound in enumerate(iteration_bounds):
            zigzag_description["loop_dim_size"][f"d{i}"] = bound

        # extract operand precision
        widths = []
        for op in generic_op.operands:
            assert isinstance(op.type, memref.MemRefType)
            element_type = op.type.get_element_type()
            if isinstance(element_type, IntegerType):
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

        print(generic_op.indexing_maps.data[0].data.results[0])
        # padding (use default of 0 padding for now)
        # affects last two indices of input I
        zigzag_description["padding"] = dict()
        workload = dict()
        workload[0] = zigzag_description

        with open("workload.py", "w") as f:
            f.write(f"workload = {workload}")

        # add stream id attribute to the generic op
        generic_op.attributes["zigzag_stream_id"] = IntAttr(0)


class LinalgToStream(ModulePass):
    name = "linalg-to-stream"

    def apply(self, ctx: MLContext, module: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            LinalgToStreamTranslator(), apply_recursively=False
        ).rewrite_module(module)
