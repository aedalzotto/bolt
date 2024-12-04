from json import load
from .tree import Tree

def bolt():
    fname = "teste.json"
    with open(fname) as f:
        model = load(f)

    feature_types = [x if x != "i" else "bool" for x in model["learner"]["feature_types"]]
    feature_names = model["learner"]["feature_names"]

    trees = model["learner"]["gradient_booster"]["model"]["trees"]

    mname = fname.split(".")[0]
    res  = "#include <stdbool.h>\n\n"
    res += "float {}({})\n{{".format(mname, ", ".join(["{} {}".format(feature_types[i], fname) for i, fname in enumerate(feature_names)]))
    for tree in trees:
        t = Tree(
            feature_types,
            feature_names, 
            tree
        )
        res += t.gen()+"\n"
    res += "\treturn {} + {};\n".format(0.5, " + ".join("w{}".format(i) for i in range(len(trees))))
    res += "}}\n".format()

    with open("{}.c".format(mname), "w") as f:
        f.write(res)
