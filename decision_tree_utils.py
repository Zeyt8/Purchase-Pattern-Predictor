from collections import Counter
from copy import deepcopy
from math import log2
import numpy as np

class Node:
    """ Representation for a node from the decision tree """
    def __init__(self, label):
        """
            for non-leafs it is the name of the attribute
            for leafs it is the class
        """
        self.label = label
        
        # Dictionary of (attribute value, nodes)
        self.children = {}
    
    def display(self, indent = ""):
        print(indent + (self.label + ":" if self.children else "<" + self.label + ">"))
        indent += "   "
        if self.children:
            for key, value in self.children.items():
                print(indent + ":" + key)
                value.display(indent + "   ")

class MyDecisionTreeClassifier:

    def __init__(self, df, T):
        self.attributes = [a for a in df.columns if a != "Revenue"]
        self.classes = np.unique(T)

    def fit(self, X, T):
        self.tree = self.__id3(X, T, self.attributes, 3)

    def evaluate(self, example):
        if self.tree.children == {}:
            return self.tree.label
        else:
            attr = self.tree.label
            val = example[attr]
            if val not in self.tree.children:
                return None
            return self.evaluate(self.tree.children[val], example)

    def __id3(self, X, T, A, d = 6):
        if len(set([t for t in T])) == 1:
            return Node(T[0])
        if len(A) == 0 or d == 0:
            return Node(self.__mostFrequentClass(T))
        A_ = deepcopy(A)
        a_star = max(A_, key=lambda a: self.__gain(X, T, a))
        A_.remove(a_star)
        n = Node(a_star)
        for val in set([x[a_star] for x in X]):
            X_val = [x for x in X if x[a_star] == val]
            T_val = [t for t, x in zip(T, X) if x[a_star] == val]
            n.children[val] = self.__id3(X_val, T_val, A_, d - 1)
        return n

    def __mostFrequentClass(self, T):
        c: Counter = Counter()
        for t in T:
            c.update([t])
        return c.most_common(1)[0][0]

    def __entropy(self, T):
        entropy  = 0
        for c in self.classes:
            p = len([t for t in T if t == c]) / len(T)
            if p != 0:
                entropy += -p * log2(p)
        return entropy

    def __gain(self, X, T, a):
        gain = self.__entropy(T)
        for val in set([x[a] for x in X]):
            X_val = [x for x in X if x[a] == val]
            gain -= len(X_val) / len(X) * self.__entropy(X_val)
        return gain