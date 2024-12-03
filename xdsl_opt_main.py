import argparse
from collections.abc import Sequence

from xdsl.xdsl_opt_main import xDSLOptMain
from xdsl.ir import MLContext

from transforms.linalg_to_stream import LinalgToStream

class xDSLOptMainWrapper(xDSLOptMain):
     
     def __init__(
        self,
        description: str = "xDSL modular optimizer driver",
        args: Sequence[str] | None = None,
    ):
        self.available_frontends = {}
        self.available_passes = {}
        self.available_targets = {}

        self.ctx = MLContext()
        super().register_all_dialects()
        super().register_all_frontends()
        super().register_all_passes()
        super().register_all_targets()

        ## Add custom dialects & passes
        super().register_pass(LinalgToStream.name, lambda: LinalgToStream())

        # arg handling
        arg_parser = argparse.ArgumentParser(description=description)
        super().register_all_arguments(arg_parser)
        self.args = arg_parser.parse_args(args=args)
        self.ctx.allow_unregistered = self.args.allow_unregistered_dialect

        super().setup_pipeline()

def main():
    xDSL = xDSLOptMainWrapper()
    xDSL.run()

if __name__ == '__main__':
    main()
