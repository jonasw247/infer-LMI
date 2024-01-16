#%%
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import os
import main_LMI_inference
import ants
import time

def runForPatient(pat):

    start_time = time.time()

    # save the parameters and the tumor
    patientNumber = ("000000" + str(pat))[-3:]
    print("patient number: ", patientNumber)

    patientPath = "/mnt/8tb_slot8/jonas/datasets/ReSPOND/respond/respond_tum_"+ patientNumber+"/d0/"
    resultPath = '/mnt/8tb_slot8/jonas/workingDirDatasets/ReSPOND/resultsLMI-inferNewVersion/' + str(patientNumber) + '/'

    # read segmentation
    segmentationNiiPath = patientPath + "sub-respond_tum_"+ patientNumber+"_ses-d0_space-sri_seg.nii.gz"

    segmentation = nib.load(segmentationNiiPath).get_fdata()

    necrotic = segmentation == 1
    enhancing = segmentation == 4

    patientFlair = segmentation == 2
    patientT1 = 1.0 * necrotic + 1.0 *enhancing

    '''smallAtlas = nib.load("/home/jonas/workspace/programs/infer-LMI/Atlasfiles/modalities/atlas_small.nii.gz").get_fdata()
    plt.imshow(smallAtlas[:,80,:])
    plt.show()

    atlasPath = "./Atlasfiles"
    atlasOtherFile = atlasImg = np.load(atlasPath+'/anatomy/npzstuffData_0001.npz')["data"][:,:,:,2]
    plt.imshow(atlasOtherFile[:,80,:])
    plt.show()
    if True:
        return 0'''


    t1Path = patientPath + "sub-respond_tum_"+ patientNumber+"_ses-d0_space-sri_t1.nii.gz"
    patientAffine = nib.load(t1Path).affine
    t1 = nib.load(t1Path).get_fdata()

    predictedTumorPatientSpace, parameterDir, referenceBackTransformed = main_LMI_inference.runLMI(t1, 1.0 *patientFlair, 1.0* patientT1, registrationMode = "t1")

    end_time = time.time()

    execution_time_in_minutes = (end_time - start_time) / 60

    print(f"The execution time in minutes is {execution_time_in_minutes}")

    parameterDir["runtime"] = execution_time_in_minutes
    os.makedirs(resultPath, exist_ok=True)
    np.save(resultPath+'lmi_parameters.npy', parameterDir)

    
    nib.save(nib.Nifti1Image(predictedTumorPatientSpace, patientAffine), resultPath+'lmi_tumor_patientSpace.nii.gz')
    nib.save(nib.Nifti1Image(referenceBackTransformed, patientAffine), resultPath+'lmi_referenceBackTransformed_patientSpace.nii.gz')
#%%
#runForPatient(1)

# %%

for i in range(2, 170):
    try:
        runForPatient(i)

    except Exception as e:
        print("error for patient ", i)
        print(e)

# %%
