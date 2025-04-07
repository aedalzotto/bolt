from argparse import ArgumentParser
from json import load
from math import log2
from .tree import Tree

def bolt():
    parser = ArgumentParser(description="Boost Learning Transpiler")
    parser.add_argument("INPUT",                    help="Input JSON")
    parser.add_argument("-o", "--output",           help="Output file   (default: INPUT.c)", default=None)
    parser.add_argument("-f", "--function",         help="Function name (default: INPUT)",   default=None)
    parser.add_argument("-q", "--quantization",     help="Quantization scalar",              default=None)
    parser.add_argument("-c", "--collapse-dummies", help="Convert dummies to labels", action="store_true")
    args = parser.parse_args()

    mname = args.INPUT.split('.')[0]
    if args.output is None:
        args.output = "{}.h".format(mname)

    if args.function is None:
        args.function = mname

    with open(args.INPUT) as f:
        model = load(f)

    base_score    = float(model["learner"]["learner_model_param"]["base_score"])
    feature_names = model["learner"]["feature_names"]
    trees         = model["learner"]["gradient_booster"]["model"]["trees"]
    feature_types = [x if x != "i" else "bool" for x in model["learner"]["feature_types"]]
    operator      = ["<" if x != "bool" else "==" for x in feature_types]
    if not args.collapse_dummies:
        for tree in trees:
            for j, idx in enumerate(tree["split_indices"]):
                if feature_types[idx] == "bool":
                    tree["split_conditions"][j] = 1
    else:
        for i, type in enumerate(feature_types):
            if type == "bool":
                feature_types[i] = "int"
                dummy = "".join(feature_names[i].split("_")[:-1])
                val   = feature_names[i].split("_")[-1]
                feature_names[i] = dummy
                for tree in trees:
                    for j, idx in enumerate(tree["split_indices"]):
                        if idx == i:
                            tree["split_conditions"][j] = val

    for tree in trees:
        for j, idx in enumerate(tree["split_indices"]):
            if feature_types[idx] == "int" and tree["left_children"][j] != -1:
                tree["split_conditions"][j] = int(tree["split_conditions"][j])
    
    res  = "#include <stdbool.h>\n\n"

    if args.quantization is not None:
        base_score = round(base_score*int(args.quantization))
        res += "int "
        for tree in trees:
            for i, child in enumerate(tree["left_children"]):
                if child == -1:
                    tree["split_conditions"][i] = round(tree["split_conditions"][i] * int(args.quantization))
    else:
        res += "float "

    res += "{}({})\n{{\n".format(args.function, ", ".join(["{} {}".format(feature_types[i], fname) for i, fname in enumerate(list(dict.fromkeys(feature_names)))]))
    for tree in trees:
        t = Tree(
            feature_types,
            feature_names, 
            "float" if args.quantization is None else "int",
            operator, 
            tree
        )
        res += t.gen()+"\n"
    res += "\treturn ({} + {})".format(base_score, " + ".join("w{}".format(i) for i in range(len(trees))))
    if args.quantization is not None:
        res += " >> {}".format(int(log2(int(args.quantization))))
    res += ";\n}\n"

    with open(args.output, "w") as f:
        f.write(res)
