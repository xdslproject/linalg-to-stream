import random
import string
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
from xdsl.dialects.builtin import ShapedType, IntegerType, IntAttr

from util.kernel_type import KernelType


class LinalgToStreamTranslator(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, generic_op: Generic, rewriter: PatternRewriter):
        # extract the kernel type from the generic op
        # as zigzag only works with MAC, MUL, ...
        kernel_type = KernelType.get_kernel(generic_op)

        if not kernel_type:
            return

        # make some assertions correct outputs of the linalg generic
        assert len(generic_op.outputs) == 1
        generic_op.outputs[0]
        # linalg should have one output, and it should be a shaped type
        if len(generic_op.outputs) != 1:
            return
        if not isinstance(generic_op.outputs[0].type, ShapedType):
            return

        # extract output op and relevant indexing maps
        output_op = generic_op.outputs[0]
        output_map = generic_op.indexing_maps.data[-1].data

        # make some assertions on correct inputs of the linalg generic
        shaped_inputs = [
            (op, map.data)
            for (op, map) in zip(generic_op.inputs, generic_op.indexing_maps.data[:-1])
            if isinstance(op.type, ShapedType)
        ]
        if len(shaped_inputs) != 2:
            return

        # input a, input b, output
        operands = [shaped_inputs[0][0], shaped_inputs[1][0], output_op]
        indexing_maps = [shaped_inputs[0][1], shaped_inputs[1][1], output_map]

        zigzag_description = dict()
        dimension_relations = []

        # for now, set operator to default type
        zigzag_description["operator_type"] = "default"

        # construct equation
        output_access = "O"
        for i in range(len(indexing_maps[-1].results)):
            map = indexing_maps[-1].results[i]
            assert isinstance(map, AffineDimExpr)
            output_access += f"[{str(map)}]"

        input_i_access = "I"
        for i in range(len(indexing_maps[0].results)):
            map = indexing_maps[0].results[i]
            if isinstance(map, AffineBinaryOpExpr):
                random_suffix = ''.join(
                    random.choice(string.ascii_letters) for _ in range(1)
                )
                dimension_relations.append(
                    f"i{random_suffix}="
                    + str(map.lhs)
                    + str(map.kind.get_token())
                    + str(map.rhs)
                )
                input_i_access += f"[i{random_suffix}]"
            else:
                assert isinstance(map, AffineDimExpr)
                input_i_access += f"[{str(map)}]"

        input_w_access = "W"
        for i in range(len(indexing_maps[1].results)):
            map = indexing_maps[1].results[i]
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

        # extract dimension_relations
        # for matmul, this is empty
        zigzag_description["dimension_relations"] = dimension_relations

        # extract loop bounds by evaluating the inverse affine map
        # with the memref shapes as input
        # results = []

        num_dims = max([dim.data.num_dims for dim in generic_op.indexing_maps.data])

        results = tuple(AffineExpr.dimension(i) for i in range(num_dims))
        combined_affine_map = AffineMap(num_dims, 0, results)
        inverse_map = combined_affine_map.inverse_permutation()

        memref_shapes = [
            shape.data for op in generic_op.operands for shape in op.type.shape.data
        ]
        iteration_bounds = inverse_map.eval(memref_shapes[:num_dims], [])
        zigzag_description["loop_dim_size"] = dict()

        for i, bound in enumerate(iteration_bounds):
            zigzag_description["loop_dim_size"][f"D{i}"] = bound

        # extract operand precision
        widths = []
        for op in operands:
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
