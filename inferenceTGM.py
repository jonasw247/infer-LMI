#%%
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import os
import main_LMI_inference
import ants
import time


start_time = time.time()

#%% save the parameters and the tumor
patientNumber = "001"
print("patient number: ", patientNumber)

patientPath = "/mnt/8tb_slot8/jonas/datasets/TGM/tgm/tgm"+ patientNumber+"/preop/"
resultPath = '/mnt/8tb_slot8/jonas/datasets/TGM/resultsLMI-infer/' + str(patientNumber) + '/'
os.makedirs(resultPath, exist_ok=True)


#%% read segmentation
segmentationNiiPath = patientPath + "sub-tgm"+ patientNumber+"_ses-preop_space-sri_seg.nii.gz"

segmentation = nib.load(segmentationNiiPath).get_fdata()

necrotic = segmentation == 4
enhancing = segmentation == 1

patientFlair = segmentation == 2
patientT1 = 1.0 * necrotic + 1.0 *enhancing

#%% generate WM mask
t1IMGPath = patientPath + "sub-tgm"+ patientNumber+"_ses-preop_space-sri_t1.nii.gz"
t1Array = nib.load(t1IMGPath).get_fdata()
patientWMAffine = nib.load(t1IMGPath).affine

brainmask = t1Array > 0

img = ants.from_numpy(t1Array)

mask = ants.from_numpy((brainmask * 1.0))# - (segmentation > 0))

print("- start tissue segmentation")
atropos = ants.atropos(a=img, m = '[0.2,1x1x1]', c = '[5,0]', i='kmeans[3]', x=mask)
print("- end tissue segmentation")
#%%
print("unique",np.unique(segmentation))

patientGM = atropos['probabilityimages'][1].numpy()
patientWM = atropos['probabilityimages'][2].numpy() 
patientWM[segmentation >0] = 1
patientGM[segmentation >0] = 0

patientWM_GM  = 0.2 * patientWM + 0.1 * patientGM 

#%%
plt.imshow(patientWM[:,:,80])
nib.save(nib.Nifti1Image(patientWM, patientWMAffine), resultPath+'_wm.nii.gz')
nib.save(nib.Nifti1Image(patientGM, patientWMAffine), resultPath+'_gm.nii.gz')
#%%


predictedTumorPatientSpace, parameterDir, wmBackTransformed = main_LMI_inference.runLMI(patientWM_GM, 1.0 *patientFlair, 1.0* patientT1,registrationMode = "WM_GM")

end_time = time.time()

execution_time_in_minutes = (end_time - start_time) / 60

print(f"The execution time in minutes is {execution_time_in_minutes}")

parameterDir["runtime"] = execution_time_in_minutes
np.save(resultPath+'lmi_parameters.npy', parameterDir)

nib.save(nib.Nifti1Image(predictedTumorPatientSpace, patientWMAffine), resultPath+'lmi_tumor_patientSpace.nii')
nib.save(nib.Nifti1Image(wmBackTransformed, patientWMAffine), resultPath+'lmi_wm_patientSpace.nii')

# %%
