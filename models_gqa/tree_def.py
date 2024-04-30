from __future__ import print_function

import errno
import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn


class ArbitraryTree(object):
    def __init__(self, idx, im_idx=-1, is_root=False):
        self.index = int(idx)
        self.is_root = is_root
        self.children = []
        self.im_idx = int(im_idx) # which image it comes from
        self.parent = None

    def add_child(self, child):
        child.parent = self
        self.children.append(child)

    def print(self):
        print('====================')
        print('is root: ', self.is_root)
        print('index: ', self.index)
        print('num of child: ', len(self.children))
        for node in self.children:
            node.print()
    
    def find_node_by_index(self, index, result_node):
        if self.index == index:
            result_node = self
        elif len(self.children) > 0:
            for i in range(len(self.children)):
                result_node = self.children[i].find_node_by_index(index, result_node)
                
        return result_node

    def search_best_insert(self, matrix_score, insert_node, best_score, best_depend_node, best_insert_node):
        # virtual node will not be considerred
        if self.is_root:
            pass
        elif float(matrix_score[self.index, insert_node.index]) > float(best_score):
            best_score = matrix_score[self.index, insert_node.index]
            best_depend_node = self
            best_insert_node = insert_node
        
        # iteratively search child
        for i in range(self.get_child_num()):
            best_score, best_depend_node, best_insert_node = \
                self.children[i].search_best_insert(matrix_score, insert_node, best_score, best_depend_node, best_insert_node)

        return best_score, best_depend_node, best_insert_node

    def get_child_num(self):
        return len(self.children)
    
    def get_total_child(self):
        sum = 0
        num_current_child = self.get_child_num()
        sum += num_current_child
        for i in range(num_current_child):
            sum += self.children[i].get_total_child()
        return sum

