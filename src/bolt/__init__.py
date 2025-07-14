from argparse import ArgumentParser
from .bolt import Bolt

def bolt():
    parser = ArgumentParser(description="Boost Learning Transpiler")
    parser.add_argument("INPUT",                       help="Input JSON")
    parser.add_argument("-o", "--output",              help="Output file   (default: INPUT.c)",                        default=None)
    parser.add_argument("-f", "--function",            help="Function name (default: INPUT)",                          default=None)
    parser.add_argument("-q", "--quantize-leaves",     help="Leaf quantization scalar (power of 2)",                   default=None)
    parser.add_argument("-c", "--collapse-dummies",    help="Convert dummies to labels",                               action="store_true")
    parser.add_argument("-m", "--minimize-int",        help="Minimize integer sizes",                                  action="store_true")
    parser.add_argument("-l", "--linear-quantization", help="Quantize features using offset (implies --minimize-int)", action="store_true")
    parser.add_argument("-r", "--rodata-conditions",   help="Place split conditions in rodata",                        action="store_true")
    args = parser.parse_args()

    model_name = args.INPUT.split('.')[0]
    if args.output is None:
        args.output = "{}.h".format(model_name)

    if args.function is None:
        args.function = model_name

    bolt = Bolt(args.INPUT)

    if args.quantize_leaves is not None:
        bolt.quantize_leaves(args.quantize_leaves)

    if args.collapse_dummies:
        bolt.collapse_dummies()

    if args.minimize_int or args.linear_quantization:
        bolt.minimize_int()

    if args.linear_quantization:
        bolt.linear_quantization()
    
    bolt.generate(args.function, rodata=args.rodata_conditions)
    bolt.write(args.output)
