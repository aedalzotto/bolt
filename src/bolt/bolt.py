from json import load
from math import log2, ceil
from .tree import Tree

class Bolt:
    def __init__(self, fname):
        with open(fname) as f:
            model = load(f)

        self.base_score    = float(model["learner"]["learner_model_param"]["base_score"])
        self.feature_names = model["learner"]["feature_names"]
        self.feature_types = [x if x != "i" else "bool" for x in model["learner"]["feature_types"]]
        self.internal_type = [x if x != "i" else "bool" for x in model["learner"]["feature_types"]]
        self.offset        = [0 for x in model["learner"]["feature_types"]]
        self.operator      = ["<" if x != "bool" else "==" for x in self.feature_types]
        self.return_type   = "float"
        self.shift         = 0

        self.trees = model["learner"]["gradient_booster"]["model"]["trees"]
        for tree in self.trees:
            del tree["base_weights"]
            del tree["categories"]
            del tree["categories_nodes"]
            del tree["categories_segments"]
            del tree["categories_sizes"]
            del tree["default_left"]
            del tree["loss_changes"]
            del tree["parents"]
            del tree["split_type"]
            del tree["sum_hessian"]
            del tree["tree_param"]

            for j, idx in enumerate(tree["split_indices"]):
                if self.feature_types[idx] == "bool":
                    tree["split_conditions"][j] = 1
                elif self.feature_types[idx] == "int":
                    tree["split_conditions"][j] = int(tree["split_conditions"][j])

    def quantize_leaves(self, scalar):
        self.return_type = "int"
        self.base_score  = round(self.base_score * int(scalar))
        self.shift       = int(log2(int(scalar)))

        for tree in self.trees:
            for i, child in enumerate(tree["left_children"]):
                if child == -1:
                    tree["split_conditions"][i] = round(tree["split_conditions"][i] * int(scalar))

    def collapse_dummies(self):
        pop_idx = []
        dummy_features = {}
        idx_skew = 0
        for i, type in enumerate(self.feature_types):
            if type == "bool":
                dummy = "".join(self.feature_names[i].split("_")[:-1])
                val = int(self.feature_names[i].split("_")[-1])
                if dummy in self.feature_names:
                    idx_skew += 1
                    feature_index = dummy_features[dummy]
                    pop_idx.append(i)
                else:
                    feature_index = (i - idx_skew)
                    dummy_features[dummy] = feature_index
                    self.feature_types[i] = "int"
                    self.internal_type[i] = "int"
                    self.feature_names[i] = dummy
                
                for tree in self.trees:
                    for j, idx in enumerate(tree["split_indices"]):
                        if idx == i and tree["left_children"][j] != -1:
                            tree["split_indices"]   [j] = feature_index
                            tree["split_conditions"][j] = val

        for i in reversed(pop_idx):
            self.feature_names.pop(i)
            self.feature_types.pop(i)
            self.internal_type.pop(i)

    def minimize_int(self):
        for i, type in enumerate(self.feature_types):
            if type == "int":
                values = [tree["split_conditions"][j] for tree in self.trees for j, idx in enumerate(tree["split_indices"]) if idx == i and tree["left_children"][j] != -1]
                if len(values) > 0:
                    min_val = min(values)
                    max_val = max(values)
                    max_val = max(abs(min_val), max_val)
                    nbits = ceil(log2(max_val)) + (1 if min_val < 0 else 0)
                else:
                    min_val = 0
                    nbits = 0
                nbits = min(i for i in [64,32,16,8] if i >= nbits)
                self.internal_type[i] = "{}int{}_t".format("u" if min_val >= 0 else "", nbits)

    def linear_quantization(self):
        for i, type in enumerate(self.internal_type):
            if "int" in type and type not in ["int8_t", "uint8_t"]:
                values = [tree["split_conditions"][j] for tree in self.trees for j, idx in enumerate(tree["split_indices"]) if idx == i and tree["left_children"][j] != -1]
                min_val = min(values) - 1
                max_val = max(values)
                interval = max_val - min_val
                if interval > 0:
                    nbits = ceil(log2(interval))
                else:
                    nbits = 0
                nbits = min(i for i in [64,32,16,8] if i >= nbits)
                self.internal_type[i] = "{}int{}_t".format("u" if min_val >= 0 else "", nbits)
                self.offset[i] = min_val
                for tree in self.trees:
                    for j, idx in enumerate(tree["split_indices"]):
                        if idx == i and tree["left_children"][j] != -1:
                            tree["split_conditions"][j] = tree["split_conditions"][j] - min_val
    
    def __build_rodata(self):
        self.conditions    = {}
        for name in self.feature_names:
            self.conditions[name] = []

        for tree in self.trees:
            tree["cond_indices"] = [-1 for i in tree["split_indices"]]

            for j, idx in enumerate(tree["split_indices"]):
                if tree["left_children"][j] != -1:
                    val = tree["split_conditions"][j]
                    if val in self.conditions[self.feature_names[idx]]:
                        val_index = self.conditions[self.feature_names[idx]].index(val)
                    else:
                        self.conditions[self.feature_names[idx]].append(val)
                        val_index = len(self.conditions[self.feature_names[idx]]) - 1
                    tree["cond_indices"][j] = val_index

    def generate(self, function, rodata=False):
        self.__build_rodata()

        self.res  = "#include <stdbool.h>\n"
        self.res += "#include <stdint.h>\n\n"

        self.res += "#define MIN(a, b) ((a) < (b) ? (a) : (b))\n"
        self.res += "#define MAX(a, b) ((a) > (b) ? (a) : (b))\n"
        self.res += "#define UINT8_MIN 0\n"
        self.res += "#define UINT16_MIN 0\n"
        self.res += "#define UINT32_MIN 0\n\n"

        if rodata:
            for i, name in enumerate(self.feature_names):
                if len(self.conditions[name]) == 0:
                    continue
                self.res += "const {} cond_{}[] = {{\n".format(self.internal_type[i], name)
                for condition in self.conditions[name]:
                    self.res += "\t{},\n".format(condition)
                self.res += "};\n\n"

        self.res += "{} {}({})\n{{\n".format(
            self.return_type, 
            function, 
            ", ".join(
                [
                    "{} {}{}".format(
                        self.feature_types[i], 
                        "f" if self.feature_types[i] != self.internal_type[i] else "", 
                        fname
                    )
                    for i, fname in enumerate(list(dict.fromkeys(self.feature_names)))
                ]
            )
        )

        quantized = False
        for i in range(len(self.feature_names)):
            if self.feature_types[i] != self.internal_type[i]:
                quantized = True
                self.res += "\t{} {} = MIN(MAX({}_MIN, f{} - {}), {}_MAX);\n".format(
                    self.internal_type[i], 
                    self.feature_names[i], 
                    self.internal_type[i].split("_")[0].upper(), 
                    self.feature_names[i], 
                    self.offset[i],
                    self.internal_type[i].split("_")[0].upper()
                )

        if quantized:
            self.res += "\n"

        for tree in self.trees:
            t = Tree(
                self.feature_names, 
                self.return_type, 
                self.operator, 
                self.conditions, 
                tree
            )
            if rodata:
                self.res += t.gen_rodata()+"\n"
            else:
                self.res += t.gen_text()+"\n"

        self.res += "\treturn ({} + {}){};\n".format(self.base_score, " + ".join("w{}".format(i) for i in range(len(self.trees))), " >> {}".format(self.shift) if self.shift != 0 else "")
        self.res += "}\n"

    def write(self, output):
        with open(output, "w") as f:
            f.write(self.res)
