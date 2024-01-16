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

    plt.imshow(patientT1[:,:,80])

    smallAtlas = nib.load("Atlasfiles/modalities/atlas_small.nii.gz").get_fdata()

    '''
    # generate WM mask
    t1IMGPath = patientPath + "sub-respond_tum_"+ patientNumber+"_ses-d0_space-sri_t1.nii.gz"
    t1Array = nib.load(t1IMGPath).get_fdata()
    patientWMAffine = nib.load(t1IMGPath).affine

    brainmask = t1Array > 0

    img = ants.from_numpy(t1Array)
    mask = ants.from_numpy((brainmask * 1.0))# - (segmentation > 0))
    print("- start tissue segmentation")
    atropos = ants.atropos(a=img, m = '[0.2,1x1x1]', c = '[5,0]', i='kmeans[3]', x=mask)
    print("- end tissue segmentation")

    print("unique",np.unique(segmentation))

    patientGM = atropos['probabilityimages'][1].numpy()
    patientWM = atropos['probabilityimages'][2].numpy() 

    patientWM[segmentation >0] = 1
    patientGM[segmentation >0] = 0
    '''
    #loadtissue
    tissuePath = patientPath + "sub-respond_tum_"+ patientNumber+"_ses-d0_space-sri_tissuemask.nii.gz"
    patientWMAffine = nib.load(tissuePath).affine
    tissue = nib.load(tissuePath).get_fdata()

    
    patientWM, patientGM = segmentation *0.0, segmentation *0.0
    patientWM[tissue == 3] = 1.0
    patientGM[tissue == 2] = 1.0
    
    patientWM_GM  = 0.2 * patientWM + 0.1 * patientGM 
    #
    os.makedirs(resultPath, exist_ok=True)

    plt.imshow(patientWM_GM[:,:,80])
    nib.save(nib.Nifti1Image(patientWM, patientWMAffine), resultPath+'_wm.nii.gz')
    nib.save(nib.Nifti1Image(patientGM, patientWMAffine), resultPath+'_gm.nii.gz')
    #
    predictedTumorPatientSpace, parameterDir, wmBackTransformed = main_LMI_inference.runLMI(patientWM_GM, 1.0 *patientFlair, 1.0* patientT1,registrationMode = "WM_GM")

    end_time = time.time()

    execution_time_in_minutes = (end_time - start_time) / 60

    print(f"The execution time in minutes is {execution_time_in_minutes}")

    parameterDir["runtime"] = execution_time_in_minutes
    np.save(resultPath+'lmi_parameters.npy', parameterDir)

    nib.save(nib.Nifti1Image(predictedTumorPatientSpace, patientWMAffine), resultPath+'lmi_tumor_patientSpace.nii.gz')
    nib.save(nib.Nifti1Image(wmBackTransformed, patientWMAffine), resultPath+'lmi_wm_patientSpace.nii.gz')
#%%
runForPatient(11)

# %%
'''
for i in range(0, 170):
    try:
        runForPatient(i)

    except Exception as e:
        print("error for patient ", i)
        print(e)
'''