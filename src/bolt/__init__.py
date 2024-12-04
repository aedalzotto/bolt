from argparse import ArgumentParser
from json import load
from .tree import Tree

def bolt():
    parser = ArgumentParser(description="Boost Learning Transpiler")
    parser.add_argument("INPUT",            help="Input JSON")
    parser.add_argument("-o", "--output",   help="Output file   (default: INPUT.c)", default=None)
    parser.add_argument("-f", "--function", help="Function name (default: INPUT)",   default=None)
    args = parser.parse_args()

    mname = args.INPUT.split('.')[0]
    if args.output is None:
        args.output = "{}.c".format(mname)

    if args.function is None:
        args.function = mname

    with open(args.INPUT) as f:
        model = load(f)

    base_score    = float(model["learner"]["learner_model_param"]["base_score"])
    feature_types = [x if x != "i" else "bool" for x in model["learner"]["feature_types"]]
    feature_names = model["learner"]["feature_names"]
    trees         = model["learner"]["gradient_booster"]["model"]["trees"]

    res  = "#include <stdbool.h>\n\n"
    res += "float {}({})\n{{\n".format(args.function, ", ".join(["{} {}".format(feature_types[i], fname) for i, fname in enumerate(feature_names)]))
    for tree in trees:
        t = Tree(
            feature_types,
            feature_names, 
            tree
        )
        res += t.gen()+"\n"
    res += "\treturn {} + {};\n".format(base_score, " + ".join("w{}".format(i) for i in range(len(trees))))
    res += "}}\n".format()

    with open(args.output, "w") as f:
        f.write(res)
