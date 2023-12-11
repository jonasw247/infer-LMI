#%%
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import tools
import os
from scipy import ndimage
from simulator import runForwardSolver


#%% run LMI WM is needed for registration
def runLMI(registrationReference, patientFlair, patientT1, registrationMode = "WM"):
    atlasPath = "./Atlasfiles"

    wmTransformed, transformedTumor, registration = tools.getAtlasSpaceLMI_InputArray(registrationReference, patientFlair, patientT1, atlasPath, getAlsoWMTrafo=True)

    #%% get the LMI prediction
    prediction = np.array(tools.getNetworkPrediction(transformedTumor))[:6]

    #%% plot the prediction
    D = prediction[0]
    rho = prediction[1]
    T = prediction[2]
    x = prediction[3]
    y = prediction[4]
    z = prediction[5]

    parameterDir = {'D': prediction[0], 'rho': prediction[1], 'T': prediction[2], 'x': prediction[3], 'y': prediction[4], 'z': prediction[5]}

    # run model with the given parameters
    brainPath = os.path.abspath('./simulator/brain')
    absPath = os.path.abspath(atlasPath + '/anatomy_dat/') + '/' # the "/"c is crucial for the solver
    tumor = runForwardSolver.run(absPath, prediction, brainPath)
    np.save('tumor.npy', tumor)

    # register back to patient space
    predictedTumorPatientSpace = tools.convertTumorToPatientSpace(tumor, registrationReference, registration)
    referenceBackTransformed = tools.convertTumorToPatientSpace(wmTransformed, registrationReference, registration)

    return predictedTumorPatientSpace, parameterDir, referenceBackTransformed

if __name__ == "__main__":
    patientNumber = 1

    print("patient number: ", patientNumber)
    patientPath = "/mnt/8tb_slot8/jonas/datasets/MichalsGlioblastomaDATA/GlioblastomaDATA/P" + str(patientNumber)+ "/P"+str(patientNumber)+"/"

    #set the paths to the wm, flair and t1c segmentations
    wmSegmentationNiiPath = patientPath + 'wm.nii'
    flairSegmentationNiiPath = patientPath + 'tumor_segm_FLAIR.nii'
    t1SegmentationNiiPath = patientPath + 'tumor_segm_T1Gd.nii'

    resultPath = 'resultsTGM/' + str(patientNumber) + '/'

    os.makedirs(resultPath, exist_ok=True)
    patientWMNib = nib.load(wmSegmentationNiiPath)

    patientFlair = nib.load(flairSegmentationNiiPath).get_fdata()
    if len(patientFlair.shape) == 4:
        patientFlair = patientFlair[:,:,:,0]
    patientT1 = nib.load(t1SegmentationNiiPath).get_fdata()
    if len(patientT1.shape) == 4:
        patientT1 = patientT1[:,:,:,0]


    patientWM = patientWMNib.get_fdata()	
    patientWMAffine = patientWMNib.affine


    predictedTumorPatientSpace, parameterDir, wmBackTransformed = runLMI(patientWM, patientFlair, patientT1)

    np.save(resultPath+'lmi_parameters.npy', parameterDir)

    nib.save(nib.Nifti1Image(predictedTumorPatientSpace, patientWMAffine), resultPath+'lmi_tumor_patientSpace.nii')

    nib.save(nib.Nifti1Image(wmBackTransformed, patientWMAffine), resultPath+'lmi_wm_patientSpace.nii')
