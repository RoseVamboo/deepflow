# -*- coding:utf-8 -*-
from functools import reduce

from .operations import Operation,Variable,Placeholder

class Session(object):
    '''
    A session to compute a particular graph
    '''

    def __init__(self):
        '''
        Session构造器
        '''
        self.graph=DEFAULT_GRAPH

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        all_nodes=(self.graph.constants+self.graph.variables+
                   self.graph.placeholders+self.graph.operations+
                   self.graph.trainable_variables)
        for node in all_nodes:
            node.output_value=None

    def run(self,operation,feed_dict=None):
        '''
        compute the output of an operation
        :param operation: 将要参加计算的operation
        :type operation:`Operation`,`Variable`,`Placeholder`

        :param feed_dict:关于在Session中的placeholder和其真实值的Map
        :return:
        '''
        postorder_nodes=_get_prerequisite(operation)

        for node in postorder_nodes:
            if type(node) is Placeholder:
                node.output_value=feed_dict[node]
            else:# Operation and variable
                node.compute_output()

        return operation.output_value

def _get_prerequisite(operation):

    postorder_nodes=[]

    # Collection nodes recursively
    def postorder_traverse(operation):
        if isinstance(operation,Operation):
            for input_node in operation.input_nodes:
                postorder_traverse(input_node)
        postorder_nodes.append(operation)

    postorder_traverse(operation)

    return postorder_nodes