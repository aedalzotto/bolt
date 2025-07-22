class Tree:
    def __init__(self, feature_names, leaf_type, operator, conditions, tree, skip_null=False):
        self.feature_names    = feature_names
        self.leaf_type        = leaf_type
        self.operator         = operator
        self.conditions       = conditions
        self.id               = tree["id"]
        self.split_indices    = tree["split_indices"]
        self.split_conditions = tree["split_conditions"]
        self.base_weights     = tree["base_weights"]
        self.left_children    = tree["left_children"]
        self.right_children   = tree["right_children"]
        self.cond_indices     = tree["cond_indices"]
        self.default_left     = tree["default_left"]
        self.skip_null        = skip_null

    def gen_text(self):
        ret  = "\t{} w{};\n".format(self.leaf_type, self.id)
        ret += self.__add_node_text(0, 1)
        return ret
    
    def gen_rodata(self):
        ret  = "\t{} w{};\n".format(self.leaf_type, self.id)
        ret += self.__add_node_rodata(0, 1)
        return ret
    
    def __add_node_text(self, index, depth):
        tab  = "\t"*depth
        left = self.left_children[index]
        if left == -1:  # Right is also -1. Node is leaf.
            return "{}w{} = {};".format(
                tab,
                self.id,
                self.base_weights[index]
            )
        
        # Otherwise is split
        right = self.right_children[index]
        return "{}if ({}{} {} {}) {{\n{}\n{}}} else {{\n{}\n{}}}".format(
            tab,
            "" if self.skip_null else
            "{} {} NULL {} *".format(
                self.feature_names[self.split_indices[index]],
                "==" if self.default_left[index] else "!=",
                "||" if self.default_left[index] else "&&"
            ),
            self.feature_names[self.split_indices[index]],
            self.operator[self.split_indices[index]],
            self.split_conditions[index],
            self.__add_node_text(left, depth+1),
            tab,
            self.__add_node_text(right, depth+1),
            tab
        )
    
    def __add_node_rodata(self, index, depth):
        tab  = "\t"*depth
        left = self.left_children[index]
        if left == -1:  # Right is also -1. Node is leaf.
            return "{}w{} = {};".format(
                tab,
                self.id,
                self.base_weights[index]
            )
        
        # Otherwise is split
        right = self.right_children[index]
        return "{}if ({}{} {} cond_{}[{}]) {{\n{}\n{}}} else {{\n{}\n{}}}".format(
            tab,
            "" if self.skip_null else
            "{} {} NULL {} *".format(
                self.feature_names[self.split_indices[index]],
                "==" if self.default_left[index] else "!=",
                "||" if self.default_left[index] else "&&"
            ),
            self.feature_names[self.split_indices[index]],
            self.operator[self.split_indices[index]],
            self.feature_names[self.split_indices[index]],
            self.cond_indices[index],
            self.__add_node_rodata(left, depth+1),
            tab,
            self.__add_node_rodata(right, depth+1),
            tab
        )
