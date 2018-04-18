# -*- coding:utf-8 -*-

from queue import Queue

import numpy as np

class Operation(object):
    '''
    Operation是deepflow中所有操作的基础类
    输入后产生零个或多个节点作为输出
    顶点可以是一个操作，变量或占位符
    '''
    def __init__(self,*input_nodes,name=None):
        '''
        构造函数
        :param input_nodes:operation节点的输入节点
        :type input_nodes:`Operation`,`Variable`,`Placeholder`
        :param name:
        '''
        #operation中接收的节点
        self.input_nodes=input_nodes

        #operation的最终输出节点
        self.output_nodes=[]

        # 本operation的输出值
        self.output_value=None

        #本operation的name
        self.name=name

        #本operation 所属图
        self.graph=DEFAULT_GRAPH

        #将这个operation 节点加入到output节点中
        for node in input_nodes:
            node.output_nodes.append(self)

        #将这个operation加入到默认的计算图中
        self.graph.operations.append(self)

        def compute_output(self):
            '''
            计算并返回这个operation的输出值
            :param self:
            :return:
            '''
            raise NotImplementedError

        def compute_gradient(self,grad=None):
            '''
            计算并返回本operation的梯度
            :param self:
            :param grad:
            :return:
            '''
            return NotImplementedError

        def __add__(self, other):
            return Add(self,other)

        def __neg__(self):
            return Negative(self)

        def __sub__(self, other):
            return Add(self,Negative(other))

        def __mul__(self,other):
            return Multiply(self,other)

#
#Add operation
#
class Add(Operation):
    '''
    an adding operation
    '''

    def __init__(self,x,y,name=None):
        '''
        构造函数
        :param x:第一个输入节点
        :type x:`Operation`,`Variable`,`Placeholder`
        :param y: 第二个输入节点
        :type y:`Operation`,`Variable`,`Placeholder`
        :param name:operation 的名字
        :type name:str
        '''
        super(self.__class__,self).__init__(x,y,name=name)

    def compute_output(self):
        '''
        计算并返回相加操作的结果
        :return:
        '''
        x,y=self.input_nodes
        self.output_value=np.add(x.output_value,y.output_value)
        return self.output_value

    def compute_gradient(self,grad=None):
        '''
        计算梯度
        :param grad:
        :return:
        '''
        x,y=[node.output_value for node in self.input_nodes]

        if grad is None:
            grad=np.ones_like(self.output_value)

        grad_wrt_x=grad
        while np.ndim(grad_wrt_x)>len(np.shape(x)):
            grad_wrt_x=np.sum(grad_wrt_x,axis=0)
        for axis,size in enumerate(np.shape(x)):
            if size==1:
                grad_wrt_x=np.sum(grad_wrt_x,axis=axis,keepdims=True)

        grad_wrt_y=grad
        while np.ndim(grad_wrt_y)>len(np.shape(y)):
            grad_wrt_y=np.sum(grad_wrt_y,axis=0)
        for axis,size in enumerate(np.shape(y)):
            if size==1:
                grad_wrt_y=np.sum(grad_wrt_y,axis=axis,keepdims=True)

        return [grad_wrt_x,grad_wrt_y]

def add(x,y,name=None):
    '''
    returns x+y element-wise
    :param x:
    :param y:
    :param name:
    :return:
    '''
    return Add(x,y,name)
#
#Multiplication operation
#
class Multiply(Operation):
    '''
    Multiply operation
    '''
    def __init__(self,x,y,name=None):
        '''
        构造函数
        :param x:第一个输入节点
        :type x:`Operation`,`Variable`,`Placeholder`
        :param y: 第二个输入节点
        :type y:`Operation`,`Variable`,`Placeholder`
        :param name: the operation name
        :type name:str
        '''
        super(self.__class__,self).__init__(x,y,name=name)

    def compute_output(self):
        '''
        计算并返回Multiplication operation结果值
        :return:
        '''
        x,y=self.input_nodes
        self.output_value=np.multiply(x.output_value,y.output_value)
        return self.output_value

    def compute_gradient(self,grad=None):
        '''
        计算并返回Multiplication operation 结果值
        :param grad:
        :return:
        '''
        x,y=[node.output_value for node in self.input_nodes]
        if grad is None:
            grad=np.ones_like(self.output_value)

        grad_wrt_x=grad*y
        while np.ndim(grad_wrt_x)>len(np.shape(x)):
            grad_wrt_x=np.sum(grad_wrt_x,axis=0)
        for axis,size in enumerate(np.shape(x)):
            if size==1:
                grad_wrt_x=np.sum(grad_wrt_x,axis=axis,keepdims=True)

        grad_wrt_y=grad*x
        while np.ndim(grad_wrt_y)>len(np.shape(y)):
            grad_wrt_y=np.sum(grad_wrt_y,axis=0)
        for axis,size in enumerate(np.shape(y)):
            if size==1:
                grad_wrt_y=np.sum(grad_wrt_y,axis=axis,keepdims=True)

        return [grad_wrt_x,grad_wrt_y]

def multiply(x,y,name=None):
    '''
    return x*y element-wise
    :param x:
    :param y:
    :param name:
    :return:
    '''
    return Multiply(x,y,name)

#
#Matrix multiplication operation
#
class MatMul(Operation):
    '''
    Matrix multiplication operation
    '''
    def __init__(self,x,y,name=None):
        '''
        MatMul构造函数
        :param x: 第一个输入节点
        :type x:`Operation`,`Variable`,`Placeholder
        :param y: 第二个输入节点
        :type y:`Operation`,`Variable`,`Placeholder`
        :param name: the operation name
        '''
        super(self.__class__,self).__init__(x,y,name=name)

    def compute_output(self):
        '''
        计算并返回Multiplication operation 值
        :return:
        '''
        x,y=self.input_nodes
        self.output_value=np.dot(x.output_value,y.output_value)
        return self.output_value

    def compute_gradient(self,grad=None):
        '''
        计算并返回结果
        :param grad:
        :return:
        '''

        #获取输入值
        x,y=[node.output_value for node in self.input_nodes]

        if grad is None:
            grad=np.ones_like(self.output_value)

        dfdx=np.dot(grad,np.transpose(y))
        dfdy=np.dot(np.transpose(x,grad))

        return [dfdx,dfdy]

def matmul(x,y,name=None):
    '''
    Multiplies matrix a by matrix b
    :param x:
    :param y:
    :param name:
    :return:
    '''
    return MatMul(x,y,name)

#
#Sigmoid
#
class Sigmoid(Operation):
    '''
    Sigmoid operation
    '''
    def __init__(self,x,name=None):
        '''
        构造函数
        :param x:输入节点
        :type x:`Operation`,`Variable`,`Placeholder`
        :param name: the operation name
        '''
        super(self.__class__,self).__init__(x,name=name)

    def compute_output(self):
        '''
        计算并返回
        :return:
        '''
        x,=self.input_nodes
        self.output_value=1/(1+np.exp(-x.ouotput_value))
        return self.output_value

    def compute_gradient(self,grad=None):
        '''
        计算sigmoid的梯度
        :param grad:
        :return:
        '''
        if grad is None:
            grad=np.ones_like(self.output_value)
        return grad*self.output_value*(1-self.output_value)

def sigmoid(x,name=None):
    '''
    compute sigmoid
    :param x:
    :param name:
    :return:
    '''
    return Sigmoid(x,name=name)

#
#Logarithm operation对数操作
#
class Log(Operation):
    """
    对数操作
    """
    def __init__(self,x,name=None):
        '''
        构造函数
        :param x:输入节点
        :type x:`Operation`,`Variable`,`Placeholder`
        :param name:
        '''
        super(self.__class__,self).__init__(x,name=name)
    def compute_output(self):
        '''
        计算并返回对数函数计算的值
        :return:
        '''
        x,=self.input_nodes
        self.output_value=np.log(x.output_value)
        return self.output_value

    def compute_gradient(self,grad=None):
        '''
        计算自然对数操作的梯度
        :param grad:
        :return:
        '''
        x=self.input_nodes[0].output_value
        if grad is None:
            grad=np.ones_like(self.output_value)
        return grad*1/x

def log(x,name=None):
    '''
    计算自然对数值
    :param x:
    :param name:
    :return:
    '''
    return Log(x,name=name)

#
#Negative operation
#
class Negative(Operation):
    '''
    Negative operation
    '''
    def __init__(self,x,name=None):
        '''
        构造函数
        :param x:
        :param name:
        '''
        super(self.__class__,self).__init__(x,name=name)

    def compute_output(self):
        """
        计算并返回
        :return:
        """
        x,=self.input_nodes
        self.output_value=-x.output_value
        return self.output_value

    def compute_gradient(self,grad=None):
        if grad is None:
            grad=np.ones_like(self.output_value)
        return -grad

def negative(x,name=None):
    return Negative(x,name=name)

#
#Reduce Sum operation
#
class ReduceSum(Operation):
    '''
    Reduce sum operation
    '''
    def __init__(self,x,axis=None):
        '''
        构造器
        :param x:输入节点
        :type x:`Operation`,`Variable`,`Placeholder`
        :param axis:the dimensions to reduce ,if `None`,reduces all dimensions
        :type axis:int
        '''
        super(self.__class__,self).__init__(x)
        self.axis=axis

    def compute_output(self):
        '''
        计算并返回ReduceSum操作的值
        :return:
        '''
        x,=self.input_nodes
        self.output_value=np.sum(x.output_value,self.axis)
        return self.output_value

    def compute_gradient(self,grad=None):
        '''
        计算Reduce sum操作的梯度
        :param grad:
        :type grad: ndarray
        :return:
        '''
        input_value=self.input_nodes[0].output_value

        if grad is None:
            grad=np.ones_like(self.output_value)

        output_shape=np.array(np.shape(input_value))
        output_shape[self.axis]=1.0
        tile_scaling=np.shape(input_value)//output_shape
        grad=np.reshape(grad,output_shape)
        return np.tile(grad,tile_scaling)

def reduce_sum(x,axis=None):
    return ReduceSum(x,axis=axis)

#
#Square operation
#
class Square(Operation):
    '''
    Square operation
    '''
    def __init__(self,x,name=None):
        '''
        构造函数
        :param x:
        :param name:
        '''
        super(self.__class__,self).__init__(x,name=name)

    def compute_output(self):
        '''
        计算并返回计算的值
        :return:
        '''
        x,=self.input_nodes
        self.output_value=np.square(x.output_value)
        return self.output_value

    def compute_gradient(self,grad=None):
        '''
        计算返回值
        :param grad:
        :return:
        '''
        input_value=self.input_nodes[0].output_value

        if grad is None:
            grad=np.ones_like(self.output_value)

        return grad*np.multiply(2.0,input_value)

def square(x,name=None):
    '''
    计算值
    :param x:
    :param name:
    :return:
    '''
    return Square(x,name=name)

#
#Constant node
#
class Constant(object):
    '''
    计算图中的Constant节点
    '''
    def __init__(self,value,name=None):
        '''
        构造器
        :param value:
        :param name:
        '''
        #value
        self.value=value

        #output value of this operation in session
        self.output_value=None

        #Nodes that receive this variable node as input
        self.output_nodes=[]

        #Operation name
        self.name=name

        # add to graph
        DEFAULT_GRAPH.constants.append(self)

    def compute_output(self):
        '''
        计算并返回constant value
        :return:
        '''
        if self.output_value is None:
            self.output_value=self.value

        return self.output_value

    def __add__(self, other):
        return Add(self,other)

    def __neg__(self):
        return Negative(self)
    def __sub__(self, other):
        return Add(self,Negative(other))

    def __mul__(self, other):
        return Multiply(self,other)

def constant(value,name=None):
    '''
    创建一个constant节点
    :param value:
    :param name:
    :return:
    '''
    return Constant(value,name=name)

#
#Variable Node
#
class Variable(object):
    '''
    Variable node in computational graph
    '''
    def __init__(self,initial_value=None,name=None,trainable=True):
        '''
        构造函数
        :param initial_value: 这个variable的初始值
        :type initial_value:number or a ndarray
        :param name: name of the variable
        :param trainable:
        '''
        #variab初始值
        self.initial_value=initial_value

        #output value of this operation in session execution
        self.output_value=None

        #Nodes that receive this variable node as input
        self.output_nodes=[]

        #Variable name
        self.name=name

        #该variable所属的计算图
        self.graph=DEFAULT_GRAPH

        #将当前的图加入
        self.graph.variables.append(self)
        if trainable:
            self.graph.trainable_variables.append(self)

    def compute_output(self):
        '''
        计算并返回variable value
        :return:
        '''
        if self.output_value is None:
            self.output_value=self.initial_value
        return self.output_value

    def __add__(self, other):
        return Add(self,other)

    def __neg__(self):
        return Negative(self)

    def __sub__(self, other):
        return Add(self,Negative(other))

    def __mul__(self, other):
        return Multiply(self,other)

#
#Placeholder
#
class Placeholder(object):
    '''

    '''
    def __init__(self,name=None):
        '''
        构造函数
        :param name:
        :return:
        '''
        # output value of this operation in session execution
        self.output_value=None

        #Nodes that receive this placeholder node as input
        self.output_nodes=[]

        #placeholder node name
        self.name=name

        #Graph the placehoder node belongs to
        self.graph=DEFAULT_GRAPH

        # add to the currently active default graph
        self.graph.placeholders.append(self)

    def __add__(self, other):
        return Add(self,other)
    def __neg__(self):
        return Negative(self)
    def __sub__(self, other):
        return Add(self,Negative(other))
    def __mul__(self, other):
        return Multiply(self,other)

def placeholder(name=None):
    '''
    Inserts a placeholder for a node that will be always fed
    :param name:
    :return:
    '''
    return Placeholder(name=name)

#Function for gradients computation

def compute_gradients(target_op):
    """
    反向传播 实现 梯度
    :param target_op:
    :return:
    """
    grad_table={}

    #
    grad_table[target_op]=np.ones_like(target_op.output_value)

    queue=Queue()
    queue.put(target_op)

    visited=set()
    visited.add(target_op)

    while not queue.empty():
        node=queue.get()

        #计算梯度
        if node!=target_op:
            grads_wrt_node_output=[]

            for output_node in node.output_nodes:

                grad_wrt_output_node_output=grad_table[output_node]

                #计算当前节点的梯度
                grad_wrt_node_output=output_node.compute_gradient(grad_wrt_output_node_output)
                if len(output_node.input_nodes)>1:
                    input_node_index=output_node.input_nodes.index(node)
                    grads_wrt_node_output.append(grad_wrt_node_output[input_node_index])
                else:
                    grads_wrt_node_output.append(grad_wrt_node_output)

            #对输出节点的梯度进行求和
            tot_grad_wrt_node_output=sum(grads_wrt_node_output)
            grad_table[node]=tot_grad_wrt_node_output

        #将邻近的节点放入队列中
        if hasattr(node,'input_nodes'):
            for input_node in node.input_nodes:
                if input_node not in visited:
                    visited.add(input_node)
                    queue.put(input_node)

    return grad_table