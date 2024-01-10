"""

Example script to run Turbsim

"""

import weis
from weis.aeroelasticse.runFAST_pywrapper import runFAST_pywrapper_batch
from weis.aeroelasticse.CaseGen_General import CaseGen_General
import numpy as np
import os, platform
from weis.aeroelasticse.turbsim_util import Turbsim_wrapper
from weis.dlc_driver.dlc_generator import 

if __name__ == '__main__':