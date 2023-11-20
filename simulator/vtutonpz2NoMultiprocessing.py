import numpy as np
import scipy as sp
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import math
#import matplotlib.pyplot as plt
#from matplotlib.widgets import Slider, Button, RadioButtons
#import scipy.interpolate
from os import listdir
import glob
import os
#import h5py
from functools import partial
import argparse
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
#import pandas as pd
#from scipy.ndimage import map_coordinates, morphology, generate_binary_structure


class converter():

    def __init__(self, vtk_path, npz_path, name = ''):

        self.tumorname = name

        self.vtk_path = vtk_path
        self.npz_path = npz_path

        self.channel = ['0']#, '1', '2']
        #channel = ['0','1','2','3', '4']
        #channel = ['0']

    def read_grid_vtk(self, data):
        # Get the coordinates of nodes in the mesh
        nodes_vtk_array= data.GetPoints().GetData()
        vertices = vtk_to_numpy(nodes_vtk_array)
        #The "Velocity" field is the vector in vtk file
        numpy_array = []
        for i in self.channel:
            vtk_array = data.GetPointData().GetArray('channel'+i)
            numpy_array.append(vtk_to_numpy(vtk_array))

        return vertices, np.array(numpy_array)

    def extract_VTK(self, filename):
        # read poly data
        self.reader.SetFileName(filename)
        # reader.ReadAllVectorsOn()
        # reader.ReadAllScalarsOn()
        self.reader.Update()
        vtk_data = self.reader.GetOutput()

        vertices, numpy_array = self.read_grid_vtk(vtk_data)

        self.numpy_data[self.x, self.y, self.z, :] = numpy_array.T

        path, filename = os.path.split(filename)
        file_name = self.npz_path+filename
        file_name = file_name.replace(".vtu",".npz")
        try:
            os.makedirs(self.npz_path+os.path.split(path)[1])
        except:
            pass

        #np.savez_compressed(file_name, data=self.numpy_data)

        return self.numpy_data

    def getArray(self):
        files_cfd = []
        for filename in sorted(glob.glob(os.path.join(self.vtk_path,'*.vtu'))):
            if ('Data_0001.vtu' in filename):
                files_cfd.append(filename)

        self.reader = vtk.vtkXMLUnstructuredGridReader()
        self.reader.SetFileName(files_cfd[0]) #files_cfd[2]
        #reader.ReadAllVectorsOn()
        #reader.ReadAllScalarsOn()
        self.reader.Update()
        vtk_data = self.reader.GetOutput()

        vertices, numpy_array = self.read_grid_vtk(vtk_data)
        bounds_cfd = vtk_data.GetBounds()


        H = np.unique(vertices[:,0]).shape[0]
        W = np.unique(vertices[:,1]).shape[0]
        D = np.unique(vertices[:,2]).shape[0]
        factor = 128
        print(f"{H}, {W}, {D}")

        self.numpy_data = np.zeros((H, W, D, len(self.channel)))

        self.x, self.y, self.z = zip(*list(map(tuple, np.uint16(factor*vertices))))

        for files in files_cfd:
            self.extract_VTK(files)

        return self.numpy_data

