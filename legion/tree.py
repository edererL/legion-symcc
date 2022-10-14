""" imports """
import z3
import random
from math import inf

from legion.helper import constraint_from_string
from Legion import uct, naive


class Arm:

    def __init__(self, node):
        self.node = node

        self.reward = 0
        self.selected = 0


    def score(self, N):
        """Computes the score in a uct way"""

        if self.node.is_leaf:
            return -inf
        else:
            return uct(self.reward, self.selected, N)


    #def descr(self, N):
    #    """Describe the uct function and its parameters"""
    #    return "uct(%d, %d, %d)" % (self.reward, self.selected, N)



class Node:

    def __init__(self, target, path, pos, neg, parent=None):
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
        """Propagate reward and selected"""

        if here:
            self.here.reward += reward
            self.here.selected += selected

        self.tree.reward += reward
        self.tree.selected += selected

        if self.parent:
            self.parent.propagate(reward, selected, here=False)


    def insert(self, trace, is_complete, decls):
        """Insert new nodes into the tree"""

        base = None
        node = self

        for index in range(len(trace)):

            was_phantom = node.is_phantom
            site, target, polarity, phi = trace[index]

            if was_phantom:
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

        if is_complete:
            node.is_phantom = False
            node.is_leaf = True

        return base, node


    def sample(self):
        """Sample a node"""

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


    def select(self, bfs):
        """Select the most interesting node"""

        if self.is_phantom:
            return self            
        else:
            if self.is_leaf:
                return self
            else:
                if bfs:
                    options = [self.yes.tree, self.no.tree]
                else:
                    options = [self.here, self.yes.tree, self.no.tree]

            N = self.tree.selected
            candidates = []
            best = -inf

            for arm in options:
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
                return node.select(bfs)


    def pp_legend(self):
        """Pretty print the legend"""

        print("              local              subtree")
        print("    score  win  try      score  win  try    path")
        self.pp()


    def pp(self):
        """Pretty print the final output"""

        if not self.parent:
            # root node
            key = "*"
        elif self.is_phantom:
            # phantom node which has never been hit explicitly but we know it is there as the negation of another known node
            key = "?"
        elif self.is_leaf:
            # leaf node
            key = "$"
        else:
            # regular internal node
            key = "."

        N = self.tree.selected

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