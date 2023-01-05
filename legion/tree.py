""" imports """
import z3
import random
from math import inf
from legion.helper import constraint_from_string
from Legion import uct, naive


class Arm:
    """ This class is employed to handle the score and reward of Nodes  """

    def __init__(self, node):
        """ Initialize instances of the Arm class """
        
        self.node = node
        self.reward = 0
        self.selected = 0


    def score(self, N):
        """Compute a node's score in a uct way and retrun it"""

        if self.node.is_leaf:
            # handle leaf nodes
            return -inf
        else:
            # apply uct method
            return uct(self.reward, self.selected, N)



class Node:
    """ This class is employed to represent program states as Nodes in a tree structured space """

    def __init__(self, target, path, pos, neg, parent=None):
        """ Initialize instances of the Node class """
        
        self.site = None
        self.target = target
        self.nbytes = len(target)
        self.path = path
        self.pos = pos
        self.neg = neg
        self.parent = parent
        self.yes = None
        self.no = None
        self.sampler = None
        self.is_phantom = True
        self.is_leaf = False
        self.here = Arm(self) 
        self.tree = Arm(self)


    def propagate(self, reward, selected, here=True):
        """ Back-propagate the values of reward and selected to the remaining tree structure """

        if here:
            self.here.reward += reward
            self.here.selected += selected

        self.tree.reward += reward
        self.tree.selected += selected

        if self.parent:
            self.parent.propagate(reward, selected, here=False)


    def insert(self, trace, is_complete, decls):
        """ Insert new nodes into the tree structure and return them """

        base = None
        node = self

        for index in range(len(trace)):
            was_phantom = node.is_phantom
            site, target, polarity, phi = trace[index]

            if was_phantom:
                # Covert the path constraints in a smt2 format
                phi = constraint_from_string(phi, decls)[0]
                
                node.is_phantom = False
                node.site = site
                node.yes = Node(target, node.path + "1", node.pos + [phi], node.neg, parent=node)
                node.no = Node(target, node.path + "0", node.pos, node.neg + [phi], parent=node)

                if not base:
                    base = node

            node = node.yes if polarity else node.no

        if not base and node.is_phantom:
            base = node

        # handle phantom and leaf node heuristic
        if is_complete:
            node.is_phantom = False
            node.is_leaf = True

        return base, node


    def sample(self):
        """ Compute a input prefix that might be path presevering by use of the naive() mathod and return it """

        if not self.target:
            return b""  # no bytes to sample

        if self.sampler is None:
            solver = z3.Optimize()
            solver.add(self.pos)
            solver.add([z3.Not(phi) for phi in self.neg])
            self.sampler = naive(solver, self.target)

        try:
            sample = next(self.sampler)
            return sample
        except StopIteration:
            return None


    def select(self, dfs):
        """ Select the a promising program state by use of a well-defined score function and return it """

        if self.is_phantom:
            # handle phantom nodes
            return self            
        else:
            # handle leaf nodes
            if self.is_leaf:
                return self
            else:
                # Legion/SymCC's selection strategy
                if dfs:
                    # depth-first selection
                    options = [self.yes.tree, self.no.tree]
                else:
                    # alternative selection strategy
                    options = [self.here, self.yes.tree, self.no.tree]

            N = self.tree.selected
            candidates = []
            best = -inf

            for arm in options:
                # depict appropriate candidates by use of the score function
                cur = arm.score(N)
                if cur == best:
                    candidates.append(arm)
                    continue
                if cur > best:
                    best = cur
                    candidates = [arm]

            arm = random.choice(candidates)
            node = arm.node

            if node is self:
                return node
            else:
                return node.select(dfs)


    def pp_legend(self):
        """ Print a legend on the console """

        print("              local              subtree")
        print("    score  win  try      score  win  try    path")
        self.pp()


    def pp(self):
        """ Print the final output on the console """

        if not self.parent:
            # root node
            key = "*"
        elif self.is_phantom:
            # phantom node
            key = "?"
        elif self.is_leaf:
            # leaf node
            key = "$"
        else:
            # regular internal node
            key = "."

        N = self.tree.selected

        # print score, reward and path
        if True or key != ".":
            a = "{:7.2f} {:4d} {:4d}".format(
                self.here.score(N), self.here.reward, self.here.selected
            )
            b = "{:7.2f} {:4d} {:4d}".format(
                self.tree.score(N), self.tree.reward, self.tree.selected
            )
            print(key, a, "  ", b, "  ", self.path)

        if key != "E":
            if self.no:
                self.no.pp()
            if self.yes:
                self.yes.pp()