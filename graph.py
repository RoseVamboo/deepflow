# -*- coding:utf-8 -*-

'''
计算图定义
'''

class Graph(object):
    '''
    计算图包含所有的计算节点
    '''
    def __init__(self):
        '''
        构造函数
        '''
        self.operations,self.constants,self.placeholders=[],[],[]
        self.variables,self.trainable_variables=[],[]

    def __enter__(self):
        '''
        重新设置默认计算图
        :return:
        '''
        global DEFAULT_GRAPH
        self.old_graph=DEFAULT_GRAPH
        DEFAULT_GRAPH=self
        return self

    def __exit__(self,exc_type,exc_val,exc_tb):
        '''
        恢复默认计算图
        :param exc_type:
        :param exc_val:
        :param exc_tb:
        :return:
        '''
        global DEFAULT_GRAPH
        DEFAULT_GRAPH=self.old_graph

    def __as_default(self):
        '''
        设置该图为全局默认图
        :return:
        '''
        return self