class Tree:
    def __init__(self, feature_types, feature_names, leaf_type, operator, tree):
        self.feature_types    = feature_types
        self.feature_names    = feature_names
        self.leaf_type        = leaf_type
        self.operator         = operator
        self.id               = tree["id"]
        self.split_indices    = tree["split_indices"]
        self.split_conditions = tree["split_conditions"]
        self.left_children    = tree["left_children"]
        self.right_children   = tree["right_children"]

    def gen(self):
        ret  = "\t{} w{};\n".format(self.leaf_type, self.id)
        ret += self.__add_node(0, 1)
        return ret
    
    def __add_node(self, index, depth):
        tab  = "\t"*depth
        left = self.left_children[index]
        if left == -1:  # Right is also -1. Node is leaf.
            return "{}w{} = {};".format(
                tab,
                self.id,
                self.split_conditions[index]
            )
        
        # Otherwise is split
        right = self.right_children[index]
        return "{}if ({} {} {}) {{\n{}\n{}}} else {{\n{}\n{}}}".format(
            tab,
            self.feature_names[self.split_indices[index]],
            self.operator[self.split_indices[index]],
            self.split_conditions[index],
            self.__add_node(left, depth+1),
            tab,
            self.__add_node(right, depth+1),
            tab
        )
