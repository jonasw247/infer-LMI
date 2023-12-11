import numpy as np
import argparse
import nibabel as nib
from glob import glob
#import pickle5 as pickle
#import tensorflow as tf
import subprocess
import time
import matplotlib.pyplot as plt
import torch
#import surface_distance
torch.set_num_threads(2) #uses max. 2 cpus for inference! no gpu inference!
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
from datetime import datetime
import ants



def load_nii(path_to_nii):
    img1 = nib.load(path_to_nii)
    img  = img1.get_fdata()
    array = np.asarray(img)
    return array

#https://github.com/deepmind/surface-distance/blob/master/surface_distance/metrics.py
def dice(gt, sim, thresholds):
    result = []
    for threshold in thresholds:
        print("T: ", threshold)
        gt_thresholded = (gt >= threshold)
        sim_thresholded = (sim >= threshold)

        total = gt_thresholded.sum() + sim_thresholded.sum()
        # assert total != 0
        print("Gt, sim, total: ", gt_thresholded.sum(), sim_thresholded.sum(), total)

        intersect = (gt_thresholded & sim_thresholded).sum()
        if total != 0:
            result.append((threshold, ((2 * intersect) / total)))
        else:
            result.append((threshold, np.nan))

    return result

def dice_norm(gt, sim, thresholds):

    gt_max = gt.max()

    gt = gt/gt_max

    sim_max = sim.max()
    sim = sim/sim_max

    result = []
    for threshold in thresholds:
        print("T: ", threshold)
        gt_thresholded = (gt >= threshold)
        sim_thresholded = (sim >= threshold)

        total = gt_thresholded.sum() + sim_thresholded.sum()
        # assert total != 0
        print("Gt, sim, total: ", gt_thresholded.sum(), sim_thresholded.sum(), total)

        intersect = (gt_thresholded & sim_thresholded).sum()
        if total != 0:
            result.append((threshold, ((2 * intersect) / total)))
        else:
            result.append((threshold, np.nan))

    return result

def hausdorff(gt, sim):

    thresholds = [0.1, 0.2, 0.4, 0.8]
    spacing_mm = [1,1,1]
    percent = 95

    gt_max = gt.max()

    gt = gt/gt_max

    sim_max = sim.max()
    sim = sim/sim_max

    result = []
    for threshold in thresholds:
        print("T: ", threshold)
        gt_thresholded = (gt >= threshold)
        sim_thresholded = (sim >= threshold)

        surface_distances = surface_distance.compute_surface_distances(gt_thresholded, sim_thresholded, spacing_mm)
        hausdorff = surface_distance.compute_robust_hausdorff(surface_distances, percent)
  
        result.append((threshold, surface_distances, hausdorff))

    return result



def remove_ticks(ax):
    ax.set_xticks([]) 
    ax.set_yticks([]) 


def mean_absolute_error_includetissue(ground_truth, output, input):
    ground_truth = np.array(ground_truth)
    output = np.array(output)
    input = input
    print("Shapes Input, output, gt: ", input.shape, output.shape, ground_truth.shape)
    wm = np.ma.masked_where(np.logical_and(input[:,:,:, 2] > 0.0001, ground_truth > 0.0001), input[:, :, :, 2]) # channel 2 od glioma solver output is wm
    gm = np.ma.masked_where(np.logical_and(input[:,:,:, 3] > 0.0001, ground_truth > 0.0001), input[:, :, :, 3]) # channel 3 of glioma solver output is gm
    csf = np.ma.masked_where(np.logical_and(input[:, :,:, 4] > 0.0001, output > 0.0001), input[:, :, :, 4]) # channle 4 is csf
    # if wm.mask.sum() == 0 or gm.mask.sum() == 0 or csf.mask.sum() == 0:
    #     return None, None, None
    print(wm.mask.sum(), gm.mask.sum() == 0, csf.mask.sum())
    mae_wm = np.mean(np.abs(output[wm.mask].ravel() - ground_truth[wm.mask].ravel()))
    mae_gm = np.mean(np.abs(output[gm.mask].ravel() - ground_truth[gm.mask].ravel()))
    mae_csf = np.mean(np.abs(output[csf.mask].ravel() - ground_truth[csf.mask].ravel()))
    return (mae_wm, mae_gm, mae_csf)



def mean_absolute_error_includetissue(ground_truth, output, input):
    ground_truth = np.array(ground_truth)
    output = np.array(output)
    input = input
    print("Shapes Input, output, gt: ", input.shape, output.shape, ground_truth.shape)
    wm = np.ma.masked_where(np.logical_and(input[:,:,:, 2] > 0.0001, ground_truth > 0.0001), input[:, :, :, 2]) # channel 2 od glioma solver output is wm
    gm = np.ma.masked_where(np.logical_and(input[:,:,:, 3] > 0.0001, ground_truth > 0.0001), input[:, :, :, 3]) # channel 3 of glioma solver output is gm
    csf = np.ma.masked_where(np.logical_and(input[:, :,:, 4] > 0.0001, output > 0.0001), input[:, :, :, 4]) # channle 4 is csf
    # if wm.mask.sum() == 0 or gm.mask.sum() == 0 or csf.mask.sum() == 0:
    #     return None, None, None
    print(wm.mask.sum(), gm.mask.sum() == 0, csf.mask.sum())

    mre_wm = tf.keras.metrics.mean_relative_error(ground_truth[wm.mask], output[wm.mask], ground_truth[wm.mask])
    mre_gm = tf.keras.metrics.mean_relative_error(ground_truth[gm.mask], output[gm.mask], ground_truth[gm.mask])
    mre_csf = tf.keras.metrics.mean_relative_error(ground_truth[csf.mask], output[csf.mask], ground_truth[csf.mask])

    return (mae_wm, mae_gm, mae_csf)


def mean_absolute_error_includetissue_masseffect(ground_truth, output, wm, gm, csf):
    ground_truth = np.array(ground_truth)
    output = np.array(output)
    print("Shapes Input, output, gt: ", wm.shape, output.shape, ground_truth.shape)
    wm = np.ma.masked_where(np.logical_and(wm > 0.0001, ground_truth > 0.0001), wm) # channel 2 od glioma solver output is wm
    gm = np.ma.masked_where(np.logical_and(gm > 0.0001, ground_truth > 0.0001), gm ) # channel 3 of glioma solver output is gm
    csf = np.ma.masked_where(np.logical_and(csf > 0.0001, output > 0.0001), csf) # channle 4 is csf
    # if wm.mask.sum() == 0 or gm.mask.sum() == 0 or csf.mask.sum() == 0:
    #     return None, None, None
    print(wm.mask.sum(), gm.mask.sum(), csf.mask.sum())
    mae_wm = np.mean(np.abs(output[wm.mask].ravel() - ground_truth[wm.mask].ravel()))
    mae_gm = np.mean(np.abs(output[gm.mask].ravel() - ground_truth[gm.mask].ravel()))
    mae_csf = np.mean(np.abs(output[csf.mask].ravel() - ground_truth[csf.mask].ravel()))
    return (mae_wm, mae_gm, mae_csf)


def mean_relative_error_includetissue_masseffect(ground_truth, output, wm, gm, csf):
    ground_truth = np.array(ground_truth)
    output = np.array(output)
    print("Shapes Input, output, gt: ", wm.shape, output.shape, ground_truth.shape)
    wm = np.ma.masked_where(np.logical_and(wm > 0.0001, ground_truth > 0.0001), wm) # channel 2 od glioma solver output is wm
    gm = np.ma.masked_where(np.logical_and(gm > 0.0001, ground_truth > 0.0001), gm ) # channel 3 of glioma solver output is gm
    csf = np.ma.masked_where(np.logical_and(csf > 0.0001, output > 0.0001), csf) # channle 4 is csf
    # if wm.mask.sum() == 0 or gm.mask.sum() == 0 or csf.mask.sum() == 0:
    #     return None, None, None
    print(wm.mask.sum(), gm.mask.sum(), csf.mask.sum())
    mre_wm = np.array((np.abs(output[wm.mask].ravel() - ground_truth[wm.mask].ravel()))/output[wm.mask].ravel())
    mre_gm = np.array((np.abs(output[gm.mask].ravel() - ground_truth[gm.mask].ravel()))/output[gm.mask].ravel())
    mre_csf = np.array((np.abs(output[csf.mask].ravel() - ground_truth[csf.mask].ravel()))/output[csf.mask].ravel())

    mre_wm = mre_wm[np.isfinite(mre_wm)]
    mre_gm = mre_gm[np.isfinite(mre_gm)]
    mre_csf = mre_csf[np.isfinite(mre_csf)]
    mre_wm = mre_wm[~np.isnan(mre_wm)]
    mre_gm = mre_gm[~np.isnan(mre_gm)]
    mre_csf = mre_csf[~np.isnan(mre_csf)]

    # mre_wm = tf.keras.metrics.MeanRelativeError(ground_truth[wm.mask], output[wm.mask], ground_truth[wm.mask])
    # mre_gm = tf.keras.metrics.MeanRelativeError(ground_truth[gm.mask], output[gm.mask], ground_truth[gm.mask])
    # mre_csf = tf.keras.metrics.MeanRelativeError(ground_truth[csf.mask], output[csf.mask], ground_truth[csf.mask])

    return (np.mean(mre_wm), np.mean(mre_gm), np.mean(mre_csf))



def plot_tumors(gt, sim, mriscan,i, dice):

    meanvalues = np.mean(mriscan, axis=(0,1))
    s = np.argmax(meanvalues)
    showparameters = 0

    for l in range(3):
            if l == 1:
                s -= 10
            elif l == 2:
                s += 20

            fig, (ax, ax2, ax3,cax2) = plt.subplots(ncols=4,figsize=(20,4), 
                    gridspec_kw={"width_ratios":[1,1,1,0.05]})
            fig.subplots_adjust(wspace=0.2)
            im = ax.imshow(mriscan[:,:,s,0].T)
            ax.set_title("MRI segmentations")
            remove_ticks(ax)
            im2 = ax2.imshow(sim[:,:,s].T)
            ax2.set_title("Simulated tumor\n Dice = {}".format(dice))
            if showparameters:
                ax2.set_title("Simulated tumor with\nDw={}, rho={}, T={}\n l={},mu1={},mu2={}".format(round(Dw,3),round(rho,3), round(T),round(l,3),round(mu1,3),round(mu2,3)))
            remove_ticks(ax2)
            im3 = ax3.imshow(gt[:,:,s].T)
            ax3.set_title("Ground truth tumor")
            remove_ticks(ax3)
            fig.colorbar(im3, cax=cax2)
            fig.savefig( "plots_masseffect/" + "_plot{}_{}.png".format(i,l))



def conv3x3_biased(in_planes, out_planes, stride=1, padding=1):
    return torch.nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, bias=True)

class BasicBlockInv_Pool_constant_noBN_n4_inplace(torch.nn.Module):
    def __init__(self, inplanes, downsample=False):
        super(BasicBlockInv_Pool_constant_noBN_n4_inplace, self).__init__()

        self.downsample = downsample
        if self.downsample:
            self.maxpool1 = torch.nn.MaxPool3d(kernel_size=2, stride=2)

        #self.bn1 = torch.nn.BatchNorm3d(inplanes)
        self.conv1 = conv3x3_biased(inplanes, inplanes)
        self.relu1 = torch.nn.ReLU(inplace=True)

        #self.bn2 = torch.nn.BatchNorm3d(inplanes)
        self.conv2 = conv3x3_biased(inplanes, inplanes)
        self.relu2 = torch.nn.ReLU(inplace=True)

        self.conv3 = conv3x3_biased(inplanes, inplanes)
        self.relu3 = torch.nn.ReLU(inplace=True)

        self.conv4 = conv3x3_biased(inplanes, inplanes)
        self.relu4 = torch.nn.ReLU(inplace=True)


    def forward(self, x):

        if self.downsample:
            x = self.maxpool1(x)

        #out = self.bn1(x)
        out = self.conv1(x)
        out = self.relu1(out)

        #out = self.bn2(out)
        out = self.conv2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.relu3(out)

        out = self.conv4(out)
        out = self.relu4(out)

        out = out + x
        #out += x

        return out

class NetConstant_noBN_l4_inplacefull(torch.nn.Module):
    def __init__(self, block, layers, numoutputs, channels, mri=False):
        super(NetConstant_noBN_l4_inplacefull, self).__init__()

        if not mri:
            self.inplanes = 2  # initial number of channels
        else:
            self.inplanes = 1

        self.conv1_i = torch.nn.Conv3d(self.inplanes, channels, kernel_size=7, stride=2, padding=2, bias=True)
        self.relu1_i = torch.nn.ReLU(inplace=True)
        self.inplanes = channels

        self.layer1 = self._make_layer(block, layers[0], downsample=False)
        self.layer2 = self._make_layer(block, layers[1])
        self.layer3 = self._make_layer(block, layers[2])
        self.layer4 = self._make_layer(block, layers[3])
        #self.layer5 = self._make_layer(block, layers[4])

        #self.bn_final = torch.nn.BatchNorm3d(channels)
        #self.relu_final = torch.nn.ReLU()
        self.avgpool = torch.nn.AdaptiveAvgPool3d((1, 1, 1))
        #self.do = torch.nn.Dropout(p=dropoutrate)
        self.fc = torch.nn.Linear(channels, numoutputs)
        #self.tanh = torch.nn.Tanh()

        # TODO: try 'fan_out' init
        for m in self.modules():
            if isinstance(m, torch.nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, torch.nn.BatchNorm3d):
                #torch.nn.init.constant_(m.weight, 1)
                #torch.nn.init.constant_(m.bias, 0)
                raise Exception("no batchnorm")
            # elif isinstance(m, torch.nn.Linear):
            #    print("initializing linear")
            #    torch.nn.init.kaiming_uniform_(m.weight, a=1.0)

    def _make_layer(self, block, blocks, downsample=True):
        layers = []
        layers.append(block(self.inplanes, downsample))
        for _ in range(1, blocks):
            layers.append(block(self.inplanes))

        return torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1_i(x)
        x = self.relu1_i(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        #x = self.layer5(x)

        #x = self.bn_final(x)
        #x = self.relu_final(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        #x = self.do(x)
        # print(f"Layer before fc: {x.mean()}, {x.std()}")
        x = self.fc(x)
        # print(f"Layer after fc: {x.mean()}, {x.std()}")
        # print("Before tanh: " + str(x))
        #x = self.tanh(x)
        # print(f"Layer after tanh: {x.mean()}, {x.std()}")

        return x

def NetConstant_noBN_64_n4_l4_inplacefull(numoutputs, mrionly=False):
        return NetConstant_noBN_l4_inplacefull(BasicBlockInv_Pool_constant_noBN_n4_inplace, [1,1,1,1], numoutputs, 64, mrionly)


def convert(mu1, mu2, x, y, z, selectedTEnd = 100): 

    T = selectedTEnd
    normalization_range = [-1.0, 1.0]

    mu1 = np.interp(mu1, normalization_range, [np.sqrt(0.01), np.sqrt(22.5)]) 
    mu2 = np.interp(mu2, normalization_range, [np.sqrt(0.1), np.sqrt(300)])  

    x = np.interp(x, normalization_range, [0.15, 0.7])
    y = np.interp(y, normalization_range, [0.2, 0.8])
    z = np.interp(z, normalization_range, [0.15, 0.7])

    D = mu1**2 / T
    rho =  mu2**2 / T 

    return D, rho, T, x, y, z , mu1, mu2


# register the pateints WM onto the atlas WM
def getAtlasSpaceLMI_InputArray(registrationReference, flairSeg, T1Seg, atlasPath, getAlsoWMTrafo = False, registrationMode = "WM"):
    print("start forward registration")

    #LMI works on Atlas where axis 1 is flipped
    antsWMPatient = ants.from_numpy(np.flip(registrationReference, axis=1))
    antsFlairPatient = ants.from_numpy(np.flip(flairSeg, axis=1))
    antsT1Patient = ants.from_numpy(np.flip(T1Seg, axis=1))

    if registrationMode == "WM": #register only with WM
        atlasImg = np.load(atlasPath+'/anatomy/npzstuffData_0001.npz')["data"][:,:,:,2]
    elif registrationMode == "WM_GM": # register with WM and GM
        atlasImg = np.load(atlasPath+'/anatomy/npzstuffData_0001.npz')["data"][:,:,:,1]
    else:
        raise Exception("registration mode not known")
    
    #masks... 
    #atlasallTissue =1.0* np.load(atlasPath+'/anatomy/npzstuffData_0001.npz')["data"][:,:,:,1]
    #atlasallTissue[atlasallTissue > 0.0001] = 1.0
    #atlasallTissue[atlasallTissue < 0.0001] = 0.0
    #mask = ants.from_numpy(atlasallTissue)
    #movingMask = ants.from_numpy(1.0 * np.invert(antsFlairPatient.numpy()>0))

    targetRegistration = ants.from_numpy(atlasImg)
    
    reg =  ants.registration( targetRegistration, antsWMPatient, type_of_transform='SyNCC')#, mask=mask, moving_mask=movingMask)

    wmPatientTransformed = ants.apply_transforms(targetRegistration, antsWMPatient, reg['fwdtransforms'])

    flairTransformed = ants.apply_transforms(targetRegistration, antsFlairPatient, reg['fwdtransforms'], interpolator='nearestNeighbor')
    t1Transformed = ants.apply_transforms(targetRegistration, antsT1Patient, reg['fwdtransforms'], interpolator='nearestNeighbor')

    tumorTransformed = 0.3333 * (flairTransformed.numpy() > 0.5)  + 0.6666 * (t1Transformed.numpy() > 0.5)

    print("finished registration")
    if getAlsoWMTrafo:
        return wmPatientTransformed.numpy(), tumorTransformed, reg
    
    return tumorTransformed

def getNetworkPrediction(transformedTumor):
    modelWeightsPath = "./modelweights/bestval-model.pt"
    modelLMI = NetConstant_noBN_64_n4_l4_inplacefull(5,True)
    checkpoint = torch.load(modelWeightsPath, map_location=torch.device('cpu'))
    modelLMI.load_state_dict(checkpoint['model_state_dict'])
    modelLMI = modelLMI.eval()

    numpyInput = transformedTumor.astype(np.float32)
    inputLMI = torch.from_numpy(np.array([[numpyInput]]))

    with torch.set_grad_enabled(False):
            with torch.autograd.profiler.profile() as prof:
                predicted = modelLMI(inputLMI) 
                
    predicted = predicted.numpy()[0]

    convPred = convert(predicted[0], predicted[1], predicted[2], predicted[3], predicted[4])

    return convPred

def convertTumorToPatientSpace(atlasTumor, patientWM, registration):

    antsTumor = ants.from_numpy(atlasTumor)
    targetRegistration = ants.from_numpy(patientWM)

    antsPredictedTumorPatientSpace = ants.apply_transforms(targetRegistration, antsTumor, registration['invtransforms'])

    return np.flip(antsPredictedTumorPatientSpace.numpy(), axis=1)



