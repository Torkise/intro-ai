import random

import numpy as np
import pandas as pd
from PIL import Image
import time
import math

np.set_printoptions(threshold=np.inf, linewidth=300)


class Map_Obj:
    "Kode fra stabben"
    def __init__(self, task=1):
        self.start_pos, self.goal_pos, self.end_goal_pos, self.path_to_map = self.fill_critical_positions(
            task)
        self.int_map, self.str_map = self.read_map(self.path_to_map)
        self.tmp_cell_value = self.get_cell_value(self.goal_pos)
        self.set_cell_value(self.start_pos, ' S ')
        self.set_cell_value(self.goal_pos, ' G ')
        self.tick_counter = 0


        #Liste med ulike veier man kan ta
        self.paths = []



    def read_map(self, path):
        """
        Reads maps specified in path from file, converts them to a numpy array and a string array. Then replaces
        specific values in the string array with predefined values more suitable for printing.
        :param path: Path to .csv maps
        :return: the integer map and string map
        """
        # Read map from provided csv file
        df = pd.read_csv(path, index_col=None,
                         header=None)  # ,error_bad_lines=False)
        # Convert pandas dataframe to numpy array
        data = df.values
        # Convert numpy array to string to make it more human readable
        data_str = data.astype(str)
        # Replace numeric values with more human readable symbols
        data_str[data_str == '-1'] = ' # '
        data_str[data_str == '1'] = ' . '
        data_str[data_str == '2'] = ' , '
        data_str[data_str == '3'] = ' : '
        data_str[data_str == '4'] = ' ; '
        return data, data_str

    def fill_critical_positions(self, task):
        """
        Fills the important positions for the current task. Given the task, the path to the correct map is set, and the
        start, goal and eventual end_goal positions are set.
        :param task: The task we are currently solving
        :return: Start position, Initial goal position, End goal position, path to map for current task.
        """
        if task == 1:
            start_pos = [27, 18]
            goal_pos = [40, 32]
            end_goal_pos = goal_pos
            path_to_map = 'Samfundet_map_1.csv'
        elif task == 2:
            start_pos = [40, 32]
            goal_pos = [8, 5]
            end_goal_pos = goal_pos
            path_to_map = 'Samfundet_map_1.csv'
        elif task == 3:
            start_pos = [28, 32]
            goal_pos = [6, 32]
            end_goal_pos = goal_pos
            path_to_map = 'Samfundet_map_2.csv'
        elif task == 4:
            start_pos = [28, 32]
            goal_pos = [6, 32]
            end_goal_pos = goal_pos
            path_to_map = 'Samfundet_map_Edgar_full.csv'
        elif task == 5:
            start_pos = [14, 18]
            goal_pos = [6, 36]
            end_goal_pos = [6, 7]
            path_to_map = 'Samfundet_map_2.csv'

        return start_pos, goal_pos, end_goal_pos, path_to_map

    def get_cell_value(self, pos):
        return self.int_map[pos[0], pos[1]]

    def get_goal_pos(self):
        return self.goal_pos

    def get_start_pos(self):
        return self.start_pos

    def get_end_goal_pos(self):
        return self.end_goal_pos

    def get_maps(self):
        # Return the map in both int and string format
        return self.int_map, self.str_map

    def move_goal_pos(self, pos):
        """
        Moves the goal position towards end_goal position. Moves the current goal position and replaces its previous
        position with the previous values for correct printing.
        :param pos: position to move current_goal to
        :return: nothing.
        """
        tmp_val = self.tmp_cell_value
        tmp_pos = self.goal_pos
        self.tmp_cell_value = self.get_cell_value(pos)
        self.goal_pos = [pos[0], pos[1]]
        self.replace_map_values(tmp_pos, tmp_val, self.goal_pos)

    def set_cell_value(self, pos, value, str_map=True):
        if str_map:
            self.str_map[pos[0], pos[1]] = value
        else:
            self.int_map[pos[0], pos[1]] = value

    def print_map(self, map_to_print):
        # For every column in provided map, print it
        for column in map_to_print:
            print(column)

    def pick_move(self):
        """
        A function used for moving the goal position. It moves the current goal position towards the end_goal position.
        :return: Next coordinates for the goal position.
        """
        if self.goal_pos[0] < self.end_goal_pos[0]:
            return [self.goal_pos[0] + 1, self.goal_pos[1]]
        elif self.goal_pos[0] > self.end_goal_pos[0]:
            return [self.goal_pos[0] - 1, self.goal_pos[1]]
        elif self.goal_pos[1] < self.end_goal_pos[1]:
            return [self.goal_pos[0], self.goal_pos[1] + 1]
        else:
            return [self.goal_pos[0], self.goal_pos[1] - 1]

    def replace_map_values(self, pos, value, goal_pos):
        """
        Replaces the values in the two maps at the coordinates provided with the values provided.
        :param pos: coordinates for where we want to change the values
        :param value: the value we want to change to
        :param goal_pos: The coordinate of the current goal
        :return: nothing.
        """
        if value == 1:
            str_value = ' . '
        elif value == 2:
            str_value = ' , '
        elif value == 3:
            str_value = ' : '
        elif value == 4:
            str_value = ' ; '
        else:
            str_value = str(value)
        self.int_map[pos[0]][pos[1]] = value
        self.str_map[pos[0]][pos[1]] = str_value
        self.str_map[goal_pos[0], goal_pos[1]] = ' G '

    def tick(self):
        """
        Moves the current goal position every 4th call if current goal position is not already at the end_goal position.
        :return: current goal position
        """
        # For every 4th call, actually do something
        if self.tick_counter % 4 == 0:
            # The end_goal_pos is not set
            if self.end_goal_pos is None:
                return self.goal_pos
            # The current goal is at the end_goal
            elif self.end_goal_pos == self.goal_pos:
                return self.goal_pos
            else:
                # Move current goal position
                move = self.pick_move()
                self.move_goal_pos(move)
                # print(self.goal_pos)
        self.tick_counter += 1

        return self.goal_pos

    def set_start_pos_str_marker(self, start_pos, map):
        # Attempt to set the start position on the map
        if self.int_map[start_pos[0]][start_pos[1]] == -1:
            self.print_map(self.str_map)
            print('The selected start position, ' + str(start_pos) +
                  ' is not a valid position on the current map.')
            exit()
        else:
            map[start_pos[0]][start_pos[1]] = ' S '

    def set_goal_pos_str_marker(self, goal_pos, map):
        # Attempt to set the goal position on the map
        if self.int_map[goal_pos[0]][goal_pos[1]] == -1:
            self.print_map(self.str_map)
            print('The selected goal position, ' + str(goal_pos) +
                  ' is not a valid position on the current map.')
            exit()
        else:
            map[goal_pos[0]][goal_pos[1]] = ' G '

    def show_map(self, map=None):
        """
        A function used to draw the map as an image and show it.
        :param map: map to use
        :return: nothing.
        """
        # If a map is provided, set the goal and start positions
        if map is not None:
            self.set_start_pos_str_marker(self.start_pos, map)
            self.set_goal_pos_str_marker(self.goal_pos, map)
        # If no map is provided, use string_map
        else:
            map = self.str_map

        # Define width and height of image
        width = map.shape[1]
        height = map.shape[0]
        # Define scale of the image
        scale = 20
        # Create an all-yellow image
        image = Image.new('RGB', (width * scale, height * scale),
                          (255, 255, 0))
        # Load image
        pixels = image.load()

        # Define what colors to give to different values of the string map (undefined values will remain yellow, this is
        # how the yellow path is painted)
        colors = {
            ' # ': (211, 33, 45),
            ' . ': (215, 215, 215),
            ' , ': (166, 166, 166),
            ' : ': (96, 96, 96),
            ' ; ': (36, 36, 36),
            ' S ': (255, 0, 255),
            ' G ': (0, 128, 255),
            ' P ': (255, 20, 260),
            ' Q ': (223, 252, 2)
        }
        # Go through image and set pixel color for every position
        for y in range(height):
            for x in range(width):
                if map[y][x] not in colors: continue
                for i in range(scale):
                    for j in range(scale):
                        pixels[x * scale + i,
                               y * scale + j] = colors[map[y][x]]
        # Show image
        image.show()

    def color_board_best_path(self, x, y): #Endrer farge i kartet på nodene beste vei
        """
        CHanges the value of a node.
        :param x: x posistion of node
        :param y: y posistion of node
        :return: None
        """
        self.set_cell_value([x, y], ' Q ')

    def color_board_searched(self, x,y): # Endrer farge i kartet på nodene som er blitt søkt i
        """
        Changes value of nodes
        :param x: x posistion of node
        :param y: y posistion of node
        :return: None
        """
        self.set_cell_value([x, y], ' P ')


class SearchNodes:
    """

    A SearchNode object.
    x, y: coordinates
    start: if the object is the starting node
    solution: If the object is the goal node
    cost: cost of traversing the node
    g: the cost of traversing from root node to this node
    h: estimated cost of traversing from node to goal node.
    f: combinde g and h of Node
    children: List of neigbour nodes
    parent: nodes parent


    """
    def __init__(self, x, y):

        self.x = x
        self.y = y
        self.start = False
        self.solution = False
        self.cost = float('inf')
        self.g = None
        self.h = None
        self.parent = None
        self.children = []
        self.f = None



def a_star(start_node):
    """
    a_star
    The algoritms pushes a node onto the open list.
    It pops the first node in the open list, appending it to the close_nodes list and expanidng the node. Children may be added to the
    open list. (Wall nodes will not be added)
    The algoritm runs as long as the open list contains at least one node. at the start of the loop.

    The algoritm traverses trough the list of children.
    If the child is neither in the open_nodes or the closed_node, it has not yet been searched.
    The attach_and_eval method is run, and the child is added to the open_node list.
    The list is sorted by f, in accending order.

    If the child is either in open- or closed_node list, the algoritm checks if it has found a
    less costly route to the node. If it has the attach_and_eval method is launched.
    If the Node is found in the closed_nodes, we have to update the cost of the children aswell, therefor
    the propagate_path_improvment is launched.
    :param start_node
    :return:
    """

    #Initializing start_node and lists.
    start_node.h = manhattan_distance(start_node)
    start_node.g = 0
    start_node.f = start_node.h + start_node.g
    closed_nodes = []
    open_nodes = [start_node]

    while True:

        if not open_nodes:
            return False

        current_node = open_nodes.pop(0)
        searched_nodes.append(current_node)
        closed_nodes.append(current_node)
        if current_node.solution:
            return True

        generate_all_successors(current_node)

        for child in current_node.children:
            if child not in open_nodes and child not in closed_nodes:
                attach_and_eval(child, current_node)
                open_nodes.append(child)
                open_nodes.sort(key=lambda x: x.f)  # Sort by lowest estimated cost

            elif current_node.g + child.cost < child.g:  # Found a cheaper path to child_Node
                attach_and_eval(child, current_node)    #Updates g, h and f of child
                if child in closed_nodes:
                    propagate_path_improvements(current_node) #Updates cost of children in node has already been searched


def attach_and_eval(child, parent):
    """
    Attaches the node to the so-far best parent.
    The best parent is the parent with the lowest g found so far by the a_star algorithm.
    Computes new g, h and f values for node.
    :param child:
    :param parent:
    :return new cost of traversing to root to node:
    """
    child.parent = parent
    child.g = parent.g + child.cost
    child.h = manhattan_distance(child)
    child.f = child.g + child.h


def propagate_path_improvents(parent):
    """
    Traverses trough children and possible other decendents.
    If a child can improve their g by getting a new parent, the parent is updatet and
    the method is run on the children for them to update the values and parents og their own decendnets
    :param parent:
    :return:
    """
    for child in parent.children:
        if parent.g + child.cost < child.g:
            child.parent = parent
            child.g = parent.g + child.cost
            propagate_path_improvents(child)


def generate_all_successors(current_node):
    """
    Traverses list of all nodes. If node is a neighbour of current_node, it is added til current_node.children
    :param current_node:
    :return:
    """
    for node in all_nodes:
        if node.cost < float('inf'):  # Check if node is not a wall
            """
            Checks that the nodes are neighbours by using coordinates. 
            """
            if (current_node.x == node.x) and (current_node.y == node.y - 1 or current_node.y == node.y + 1):
                current_node.children.append(node)
            elif (current_node.y == node.y) and (current_node.x == node.x - 1 or current_node.x == node.x + 1):
                current_node.children.append(node)


def manhattan_distance(node):
    """
    Estimating the distance from node to goal.
    Does not consider the cost of the nodes.
    :param node:
    :return manhatten distance between node and goal:
    """
    return abs(goal_node.x - node.x) + abs(goal_node.y - node.y)




def get_all_nodes(filename):   #lager noder ved hjelp av filvanet
    """
    Creates nodes from coordinates in file.
    Returns a list of all nodes
    :param filename:
    :return list of nodes:
    """
    returning_list = []
    nodes_from_map = map_obj.get_maps()[0]
    for x_coord in range (46): #Hard coded in the limits. Will only tranverse blocks inside the list. Not apllyable to other datasets than the ones we have
        for y_coord in range(38):
            if nodes_from_map[x_coord, y_coord] == -1: #wallNode. node.cost will remaing infinate
                wall_node = SearchNodes(x_coord, y_coord)
                returning_list.append(wall_node)
            else:  #not wall node,  hardkode med dgb
                good_guy_node = SearchNodes(x_coord, y_coord)
                good_guy_node.cost = nodes_from_map[x_coord, y_coord] #setting the cost
                returning_list.append(good_guy_node)
    return returning_list

#Definign task
print("Choose an task between 1 and 4")
task = int(input())
while not (1 <= task <= 4):
    print("Please choose an task between 1 and 4")
    task = int(input())
# Part 1
# Oppretter kartobjektet
map_obj = Map_Obj(task=task)
# map_obj.show_map()

#Getting positions of critical nodes and filename
start_pos, goal_pos, end_goal, path_to_map = map_obj.fill_critical_positions(task=task)

#Creating start and end nodes with coordinates
start_node = SearchNodes(start_pos[0], start_pos[1])
goal_node = SearchNodes(goal_pos[0], goal_pos[1])
end_goal_node = SearchNodes(end_goal[0], end_goal[1])

#creating list of nodes.
all_nodes = get_all_nodes(path_to_map)

#updating states of critical nodes
for node in all_nodes:

    if node.x == start_node.x and node.y == start_node.y:
        node.start = True
        start_node = node
    elif node.x == goal_node.x and node.y == goal_node.y:
        node.solution = True
        goal_node = node
    elif node.x == end_goal_node.x and node.y == end_goal_node.y:

        node = end_goal_node


best_path = []
searched_nodes = []

map_obj.show_map()

if a_star(start_node):
    """
    Finds the path from goal node to start node, following the parents 
    """
    print("Path found")
    print("Start-node: [" + str(start_node.x) + ", " + str(start_node.y) + "]")
    print("Goal-node: [" + str(goal_node.x) + ", " + str(goal_node.y) + "]")

    best_path.append(goal_node)
    node = goal_node.parent
    while node != start_node:
        best_path.append(node)
        node = node.parent

    for node in searched_nodes:
        map_obj.color_board_searched(node.x, node.y)  #Updating value of searched nodes

    for node in best_path:
        map_obj.color_board_best_path(node.x, node.y) #Updating value of nodes in path

else:
    print("You suck, men koden har ikke kræsha så det e bra (y)")


map_obj.show_map()
