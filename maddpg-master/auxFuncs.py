from datetime import datetime
import numpy as np
import cvxpy as cp
from matplotlib import pyplot as plt
import matplotlib
import random
matplotlib.use("Qt5Agg")

# ------------------------------
#            SB 
# ------------------------------

def calculateSBOptimal(**kwargs):
    ################# GLOBAL #################
    # load = [3,1,1]
    demand = kwargs["demand"] if "demand" in kwargs else [3, 1, 1]
    timeStep=len(demand)
    demandCharge = cp.Variable(len(demand))
    ############ SCALAR VARIABLES ############
    load = cp.Variable(len(demand))

    ############## CONSTRAINTS ################
    constraints = [0 <= load, 
                    load <= max(demand)]

    ############## CALCULATIONS ###############
    deltaPenalty = (demand - load)**2

    ################ SOLVER ##################
    function = load**2 + deltaPenalty 

    objective = cp.Minimize(cp.sum(function))

    prob = cp.Problem(objective, constraints)

    prob.solve()

    # print("Solution type: ", prob.status)
    # print("-"*25)
    # print("Minimized value: ", round(prob.value, 3))
    # print("Optimal value: ", [round(i, 3) for i  in load.value])
    # print("Demand Charge: ", round(max(load.value), 3))
    
    return [round(i, 3) for i in load.value]
    
    
    
# ------------------------------
#            EV 
# ------------------------------

def calculateEVOptimal(**kwargs):

    ################# GLOBAL #################
    required = kwargs["required"] if "required" in kwargs else 2
    ############ SCALAR VARIABLES ############
    timeStep = range(3)
    load = cp.Variable(len(timeStep))
    
    ############## CALCULATIONS ###############
    print(required)
    ################ SOLVER ##################
    # function = required - load 
    function = required - sum([i for i in load])
    
    ############## CONSTRAINTS ################
    constraints = [0 <= load, 
                load <= 1,
                function >= 0]

    objective = cp.Minimize(cp.sum(function))
    
    prob = cp.Problem(objective, constraints)
    
    prob.solve()
    
    #print("Solution type: ", prob.status)
    #print("-"*25)
    #print("Minimized value: ", round(prob.value, 3))
    #print("Optimal value: ", [round(i, 3) for i  in load.value])
    #print("Demand Charge: ", round(max(load.value), 3))
    
    return [round(i,3) for i in load.value]
    
    
def calculatePerformanceSB(**kwargs):
    
    demand = kwargs["demand"] if 'demand' in kwargs else [3,1,1]

    return demand
    
    
    
def calculatePerformanceEV(**kwargs):
    
    required = kwargs["required"] if 'required' in kwargs else 2
    
    timeStep = int(required)
    
    requiredList = [(timeStep/required) for x in range(timeStep)]
    # requiredList = [(timeStep/required) for x in range(timeStep)]
    
    return requiredList