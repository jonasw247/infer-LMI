#%%
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import tools
import os
from scipy import ndimage
from simulator import runForwardSolver

atlasPath = "./Atlasfiles"

for i in range(6, 7):

    ############################################################
    ###################### settings start ######################

    patientNumber = i
    print("patient number: ", patientNumber)
    patientPath = "/mnt/8tb_slot8/jonas/datasets/MichalsGlioblastomaDATA/GlioblastomaDATA/P" + str(patientNumber)+ "/P"+str(patientNumber)+"/"

    #set the paths to the wm, flair and t1c segmentations
    wmSegmentationNiiPath = patientPath + 'wm.nii'
    flairSegmentationNiiPath = patientPath + 'tumor_segm_FLAIR.nii'
    t1SegmentationNiiPath = patientPath + 'tumor_segm_T1Gd.nii'

    resultPath = 'results/' + str(patientNumber) + '/'
    ###################### settings end #########################
    #############################################################

    os.makedirs(resultPath, exist_ok=True)

    #%% register to correct 128 file
    patientWMNib = nib.load(wmSegmentationNiiPath)
    patientWM = patientWMNib.get_fdata()	
    patientWMAffine = patientWMNib.affine

    patientFlair = nib.load(flairSegmentationNiiPath).get_fdata()
    if len(patientFlair.shape) == 4:
        patientFlair = patientFlair[:,:,:,0]
    patientT1 = nib.load(flairSegmentationNiiPath).get_fdata()
    if len(patientT1.shape) == 4:
        patientT1 = patientT1[:,:,:,0]

    wmTransformed, transformedTumor, registration = tools.getAtlasSpaceLMI_InputArray(patientWM, patientFlair, patientT1, atlasPath, getAlsoWMTrafo=True)

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
    np.save(resultPath+'lmi_parameters.npy', parameterDir)

    #plt.imshow(transformedTumor[:,:,int(z*129)], cmap='Reds', alpha=0.6)
    imageAtlas = np.load(atlasPath+'/anatomy/npzstuffData_0001.npz')["data"][:,:,:,2]
    plt.imshow(imageAtlas[:,:,int(z*129)], cmap='Greys', alpha=0.6)

    plt.imshow(transformedTumor[:,:,int(z*129)], cmap='Reds', alpha=0.6)
    plt.scatter(y*129, x*129, c='r')

    #%% run model with the given parameters
    brainPath = os.path.abspath('./simulator/brain')
    absPath = os.path.abspath(atlasPath + '/anatomy_dat/') + '/' # the "/"c is crucial for the solver
    tumor = runForwardSolver.run(absPath, prediction, brainPath)
    np.save('tumor.npy', tumor)
    #%%
    #tumor = np.load('tumor.npy')

    # %% register back to patient space

    predictedTumorPatientSpace = tools.convertTumorToPatientSpace(tumor, patientWM, registration)
    #%% save

    nib.save(nib.Nifti1Image(predictedTumorPatientSpace, patientWMAffine), resultPath+'lmi_tumor_patientSpace.nii')
