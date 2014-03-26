from pyutilib.component.core import *
from coopr.pysp import solutionwriter
from coopr.pysp.scenariotree import *
from six import iteritems
import os
import simplejson

def index_to_string(index):

    result = str(index)
    result = result.lstrip('(').rstrip(')')
    result = result.replace(',',':')
    result = result.replace(' ','')

    return result

class JSONSolutionWriter(SingletonPlugin):
 
    implements (solutionwriter.ISolutionWriterExtension)
 
    def write(self, scenario_tree, instance_dictionary, output_file_prefix):
 
        if not isinstance(scenario_tree, ScenarioTree):
            raise RuntimeError("JSONSolutionWriter write method expects ScenarioTree object - type of supplied object="+str(type(scenario_tree)))
     
        data = {}    
        for stage in scenario_tree._stages:
            stage_name = stage._name
            if stage_name not in data.keys():
                data[stage_name] = {}
            for tree_node in stage._tree_nodes:
                tree_node_name = tree_node._name
                if tree_node_name not in data[stage_name].keys():
                    data[stage_name][tree_node_name] = {}
                 
                for variable_id, (var_name, index) in iteritems(tree_node._variable_ids):
                    if var_name not in data[stage_name][tree_node_name].keys():
                        data[stage_name][tree_node_name][var_name] = {}
                     
                    if index is None:
                        data[stage_name][tree_node_name][var_name] = str(tree_node._solution[variable_id])
                    elif  index not in data[stage_name][tree_node_name][var_name].keys():
                        data[stage_name][tree_node_name][var_name][index] =  str(tree_node._solution[variable_id])
                     
        output_filename = "%s.json"%(output_file_prefix)                      
        with open(output_filename,"w") as output_file:
            output_file.write(simplejson.dumps(data))
            output_file.close()
           
        print("Scenario tree solution written to file %s"%(output_filename))

