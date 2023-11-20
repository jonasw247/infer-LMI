
#%%
import numpy as np
import os
import time
import datetime
import subprocess
import shutil
from .vtutonpz2NoMultiprocessing import converter as VtuToNpz

#from tool import generalTools as tools

#%%
def simulationRunOnDatFiles(convertedParameters, outputFolder, anatomyFolder, patientLabel = "noLabel", pathToSimulator = "./brain", doSave =False):
    #print time as string
    string = datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S") + str(np.random.randint(0,1000000))

    vtuPath = os.path.join(outputFolder, "vtus" + str(patientLabel) + string + '/')
    os.makedirs(vtuPath, exist_ok=True)
    npzPath = os.path.join(outputFolder, "npzs" + str(patientLabel) + string+'/')
    os.makedirs(npzPath, exist_ok=True)

    predDw, predRho, predTend, predIcx, predIcy, predIcz = convertedParameters

    dumpFreq = 0.9999 * predTend

    command = pathToSimulator + " -model RD -PatFileName " + anatomyFolder + " -Dw " + str(
    predDw) + " -rho " + str(predRho) + " -Tend " + str(int(predTend )) + " -dumpfreq " + str(dumpFreq) + " -icx " + str(
    predIcx) + " -icy " + str(predIcy) + " -icz " + str(predIcz) + " -vtk 1 -N 16 -adaptive 0"


    print(" ")
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print('run ', command)

    start = time.time()
    simulation = subprocess.check_output([command], shell=True, cwd=vtuPath)  # e.g. ./vtus0/sim/
    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<	")
    print(" ")

    #vtu2npz = subprocess.check_call(["python3 vtutonpz2.py --vtk_path " + vtuPath + " --npz_path " + npzPath ], shell=True)
    #vtu2npz = subprocess.check_call(["python3 vtutonpz2NoMultiprocessing.py --vtk_path " + vtuPath + " --npz_path " + npzPath ], shell=True)
    converter = VtuToNpz(vtuPath, npzPath)
    array = converter.getArray()[:,:,:,0]

    shutil.rmtree(vtuPath)

    end = time.time()
    saveDict = {}

    saveDict['predConverted'] = convertedParameters
    saveDict['simtime'] = start-end
    saveDict['anatomyFolder'] = anatomyFolder

    #array = np.load( os.path.join(npzPath, "Data_0001.npz"))['data'][:,:,:,0]

    if doSave:
        np.save( os.path.join(npzPath, "allParams.npy"), saveDict)
    else:
        shutil.rmtree(npzPath)

    return array

def run(datPath, parameter, brianPath):

    resultArray = simulationRunOnDatFiles(parameter, './tempOutput/', anatomyFolder = datPath, pathToSimulator= brianPath)
   
    return resultArray

