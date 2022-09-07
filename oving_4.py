from unicodedata import name
import graphviz
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import operator, uuid
import random


#Reading data from the csv files 
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#Testing that the reading works
print(train)
print(test)

#Getting unique values in the dataset for each row. 
def unique_values(rows, colums):
    return set([row[colums]] for row in rows)

unique_values = unique_values(train.rows, train.colums)



def plurality_value(examples):
    #Selects the most common outputvalue among a set of examples. Breaks ties random 
    pass

def same_classification(examples):
    pass

def importance(a, examples):
    pass

#Gain(A) = B(p/(p+n)) - Remainder(A)
def information_gain():
    pass

def remainder():
    pass

def decision_tree_learning(examples, attributes, parent_examples): 

    #examples is a pandas.DataFrame object 
    if examples.empty:
        # TODO: return empty 
        return plurality_value(parent_examples)
    # TODO If all examples have the same classification: 
    elif same_classification(examples):
        # TODO: return class 
        return None #return classifications 
    
    elif attributes.empty:
        # TODO: retrun mode(examples)   
        return plurality_value(examples)
    else: 
        for element in attributes:
            a = np.argmax(importance(a, examples))
            tree = graphviz.Digraph(name="Decision Tree", filename="tree_dt")
            tree.node(name="a_name", label='a')
            for value in a: 
                exs = None #{e: e element of examples and e.A = v_k}
                child_attributes = attributes
                child_attributes.remove(a)
                subtree = decision_tree_learning(exs, child_attributes, examples)
                subtree.node(name="name", label="A = v_k")
        return tree



# Example from book on how to use Graphviz 
tree = graphviz.Digraph(name="Decision Tree", filename="tree_dt")
tree.node(name ="attr1", label='atrr1')

tree.node(name='value_1', label='1')
tree.node(name='value_2', label='2')

tree.edge(tail_name="attr_1", head_name="value_1", label="2")
tree.edge(tail_name="attr_1", head_name="value_2", label="1")

tree.render(view=False)