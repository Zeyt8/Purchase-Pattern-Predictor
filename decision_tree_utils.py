from collections import Counter
from math import log2
import pandas as pd
import numpy as np
import numpy.typing as npt

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

class MyDecisionTreeClassifier:

    def __init__(self, df, Class):
        self.df = df
        self.Class = Class

    def fit(self, X, T) -> None:
        self.tree = self.__id3(self.df, self.df.drop(self.Class, axis=1).columns, 3)

    def predict(self, X) -> npt.NDArray[np.bool_]:
        results = np.empty(len(X), dtype=np.bool_)

        for i, example in enumerate(X):
            results[i] = self.__evaluate(self.tree, example)

        return results

    def __id3(self, df: pd.DataFrame, A: pd.Index, d = 6) -> Node:
        if len(df[self.Class].unique()) == 1:
            return Node(df.iloc[0][self.Class])
        if len(A) == 0 or d == 0:
            return Node(self.__mostFrequentClass(df))
        A_copy = A.copy()
        a_star = max(A_copy, key=lambda a: self.__gain(df, a))
        A_copy.drop(a_star)
        n = Node(a_star)
        for val in df[a_star].unique():
            df_val = df[df[a_star] == val]
            n.children[val] = self.__id3(df_val, A_copy, d - 1)
        return n

    def __mostFrequentClass(self, df: pd.DataFrame) -> str:
        c: Counter = Counter(df[self.Class])
        return c.most_common(1)[0][0]

    def __entropy(self, df: pd.DataFrame) -> float:
        entropy = 0
        total_records = len(df)
        for c in df[self.Class].unique():
            p = len(df[df[self.Class] == c]) / total_records
            if p != 0:
                entropy += -p * log2(p)
        return entropy

    def __gain(self, df: pd.DataFrame, a) -> float:
        gain = self.__entropy(df)
        total_records = len(df)
        for val in df[a].unique():
            df_val = df[df[a] == val]
            gain -= len(df_val) / total_records * self.__entropy(df_val)
        return gain
    
    def __evaluate(self, tree, example):
        if tree.children == {}:
            return bool(tree.label)
        else:
            attr = tree.label
            val = example[self.df.columns.get_loc(attr)]
            closest_child = min(tree.children.keys(), key=lambda x: abs(x - val))
            return self.__evaluate(tree.children[closest_child], example)