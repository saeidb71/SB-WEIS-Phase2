import numpy as np
import openmdao.api as om
import pickle
import os

mydir = os.path.dirname(os.path.realpath(__file__))  # get path to this file

pkl_path = mydir + os.sep + "outputs" + os.sep + "IEA_level2" + os.sep + "ABCD_matrices.pkl"

with open(pkl_path, 'rb') as handle:
    ABCD_list = pickle.load(handle)
    
print("Information available in the pickle file:")
for key in ABCD_list[0]:
    print(key)
print()    

sql_path = mydir + os.sep + "outputs" + os.sep + "IEA_level2" + os.sep + "log_opt.sql"

cr = om.CaseReader(sql_path)

driver_cases = cr.get_cases('driver')
breakpoint()
A_plot = []
DVs = []
for idx, case in enumerate(driver_cases):
    print('===================')
    print('Simulation index:', ABCD_list[idx]['sim_idx'])
    dvs = case.get_design_vars(scaled=False)
    for key in dvs.keys():
        print(key)
        print(dvs[key])
    print()
    print("A matrix")
    print(ABCD_list[idx]['A'])
    print()
    
    A_plot.append(ABCD_list[idx]['A'][1, 1])
    DVs.append(dvs[key])
    
# import matplotlib.pyplot as plt

# A_plot = np.array(A_plot)
# DVs = np.array(DVs)

# plt.scatter(DVs, A_plot[:])

# plt.xlabel("Tower Young's Modulus, Pa")
# plt.ylabel('A[1, 1]')
# plt.tight_layout()

# plt.show()