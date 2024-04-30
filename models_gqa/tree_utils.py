import errno
import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
from torch.autograd import Variable
import tree_def, tree_utils, tree_def
import config


def generate_tree(gen_tree_input):
    """
    """
    return generate_arbitrary_trees(gen_tree_input)


def generate_arbitrary_trees(inputpack):
    """ Generate arbiraty trees according to the bbox scores """
    scores, is_training = inputpack
    trees = []
    batch_size, num_obj, _ = scores.size()
    sym_score = scores + scores.transpose(1, 2)
    rl_loss = []
    entropy_loss = []
    for i in range(batch_size):
        slice_container, index_list = return_tree_contrainer(num_obj, i)
        slice_root_score = (sym_score[i].sum(1) - sym_score[i].diag()) / 100.0
        slice_tree = return_tree(sym_score[i], slice_root_score, slice_container, index_list, rl_loss, entropy_loss,
                                   is_training)
        adj_matrix = tree_to_adjacency_matrix(slice_tree, num_obj)
        trees.append(adj_matrix)
    # trees = [create_single_tree(sym_score[i], i, num_obj) for i in range(batch_size)]

    return trees, rl_loss, entropy_loss


# def create_single_tree(single_sym_score, i, num_obj):
#    slice_container = [tree_def.ArbitraryTree(index, im_idx=i) for index in range(num_obj)]
#    index_list = [index for index in range(num_obj)]
#    single_root_score = single_sym_score.sum(1) - single_sym_score.diag()
#    slice_bitree = ArTree_to_BiTree(return_tree(single_sym_score, single_root_score, slice_container, index_list))
#    return slice_bitree

def add_edge(matrix, src, dest):
    # 由于是无向图，我们添加从src到dest和从dest到src的边
    matrix[src][dest] = 1
    matrix[dest][src] = 1


def tree_to_adjacency_matrix(tree, obj_num):
    if not tree.children and not tree.is_root:
        raise ValueError("The tree must have at least one root node.")

    # 找出所有节点的索引
    nodes = set()

    def add_nodes(node):
        nodes.add(node.index)
        if node.children:
            for child in node.children:
                add_nodes(child)

    # 从根节点开始添加所有节点
    if tree.is_root:
        add_nodes(tree)
    else:
        # 如果传入的不是根节点，需要找到根节点
        root = None
        for node in tree.traverse():
            if node.is_root:
                root = node
                break

        if root is None:
            raise ValueError("No root node found.")

        add_nodes(root)

    # 初始化邻接矩阵
    num_nodes = len(nodes)
    assert num_nodes == obj_num
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

    # 填充邻接矩阵
    def fill_matrix(node):
        for child in node.children:
            add_edge(adjacency_matrix, node.index, child.index)
            fill_matrix(child)

    if tree.is_root:
        fill_matrix(tree)
    else:
        fill_matrix(root)

    return adjacency_matrix


def return_tree(matrix_score, root_score, node_containter, remain_list, gen_tree_loss_per_batch, entropy_loss,
                is_training):
    """ Generate An Arbitrary Tree by Scores """
    virtual_root = tree_def.ArbitraryTree(-1, im_idx=-1)
    virtual_root.is_root = True

    start_idx = int(root_score.argmax())
    start_node = node_containter[start_idx]
    virtual_root.add_child(start_node)
    assert (start_node.index == start_idx)
    select_list = []
    selected_node = []
    select_list.append(start_idx)
    selected_node.append(start_node)
    remain_list.remove(start_idx)
    node_containter.remove(start_node)

    not_sampled = True

    while (len(node_containter) > 0):
        wid = len(remain_list)

        select_index_var = Variable(torch.LongTensor(select_list).cuda())
        remain_index_var = Variable(torch.LongTensor(remain_list).cuda())
        select_score_map = torch.index_select(torch.index_select(matrix_score, 0, select_index_var), 1,
                                              remain_index_var).contiguous().view(-1)

        # select_score_map = matrix_score[select_list][:,remain_list].contiguous().view(-1)
        if config.use_rl and is_training and not_sampled:
            dist = F.softmax(select_score_map, 0)
            greedy_id = select_score_map.max(0)[1]
            best_id = torch.multinomial(dist, 1)[0]
            if int(greedy_id) != int(best_id):
                not_sampled = False
                if config.log_softmax:
                    prob = dist[best_id] + 1e-20
                else:
                    prob = select_score_map[best_id] + 1e-20
                gen_tree_loss_per_batch.append(prob.log())
            # neg_entropy = dist * (dist + 1e-20).log()
            # entropy_loss.append(neg_entropy.sum())
        else:
            _, best_id = select_score_map.max(0)
        # _, best_id = select_score_map.max(0)
        depend_id = int(best_id) // wid
        insert_id = int(best_id) % wid

        best_depend_node = selected_node[depend_id]
        best_insert_node = node_containter[insert_id]
        best_depend_node.add_child(best_insert_node)

        selected_node.append(best_insert_node)
        select_list.append(best_insert_node.index)
        node_containter.remove(best_insert_node)
        remain_list.remove(best_insert_node.index)
    if not_sampled:
        gen_tree_loss_per_batch.append(Variable(torch.FloatTensor([0]).zero_().cuda()))
    return virtual_root


def return_tree_contrainer(num_nodes, batch_id):
    """ Return number of tree nodes """
    container = []
    index_list = []
    for i in range(num_nodes):
        container.append(tree_def.ArbitraryTree(i, im_idx=batch_id))
        index_list.append(i)
    return container, index_list


def print_tree(tree):
    if tree is None:
        return
    if (tree.left_child is not None):
        print_node(tree.left_child)
    if (tree.right_child is not None):
        print_node(tree.right_child)

    print_tree(tree.left_child)
    print_tree(tree.right_child)

    return


def print_node(tree):
    print(' depth: ', tree.depth(), end="")
    print(' score: ', tree.node_score, end="")
    print(' child: ', tree.get_total_child())
