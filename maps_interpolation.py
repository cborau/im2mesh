# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 15:15:59 2022

@author: Diego
"""

import numpy as np
from scipy.interpolate import griddata
import os
import pydicom as pyd
from pathlib import Path
import nibabel as nib
from gui_options import *
from formatreader import voxel2space
import argparse


def load_data_dicom(directory):
    """
    Reads the imaging data and converts it to spatial coordinates.
    
    Parameters
    ----------
    directory : str 
        Path to the folder containing the DICOM files.

    Returns
    -------
    coords : Numpy array
        Contains the spatial coordinates of the z-stacked images.

    values : Numpy array
        Contains the parameter values for each of the points in coords array.
        
    """

    im_pos_all = np.array([])
    #Get the first dicom in the dir to get the image size
    init_file = os.listdir(directory)[0] 
    n_files = len(os.listdir(directory))
    ds_init = pyd.dcmread(Path(directory, init_file))
    img_init = ds_init.pixel_array
    img_3d = np.empty((img_init.shape[0],img_init.shape[1], n_files)) #Initialize empty 3D array with initial image size and number of images
    #Sort files by 
    first_slice = True
    for file in sorted(os.listdir(directory)):
        if file.endswith(".dcm"):
            ds = pyd.dcmread(Path(directory, file))
            img = ds.pixel_array
            slice_number = sorted(os.listdir(directory)).index(file)
            img_3d[:,:,slice_number] = img
            # Extract the necessary parameters from DICOM header
            im_pos = list(map(float, ds.ImagePositionPatient))  # Image position
            im_pos_number = np.array([slice_number] + im_pos)
            im_pos_all = np.append(im_pos_all, im_pos_number).reshape(-1, 4)
            if first_slice:
                im_pos_M = np.array(
                    ds.ImagePositionPatient)  # Image position of the first slice to compute the transformation matrix M
                im_or = np.array(ds.ImageOrientationPatient)  # Image orientation
                pix_sp = np.array(ds.PixelSpacing)  # Pixel spacing
                first_slice = False
    # Transformation matrix
    # We first compute the transformation matrix for a single slice
    # Link to source: https://nipy.org/nibabel/dicom/dicom_orientation.html#dicom-voxel-to-patient-coordinate-system-mapping
    # Image orientation values are flipped (columnwise) due to the DICOM coordinate system
    transf_mat = np.array([[im_or[3] * pix_sp[0], im_or[0] * pix_sp[1], 0, im_pos_M[0]],
                           [im_or[4] * pix_sp[0], im_or[1] * pix_sp[1], 0, im_pos_M[1]],
                           [im_or[5] * pix_sp[0], im_or[2] * pix_sp[1], 0, im_pos_M[2]], [0, 0, 0, 1]])
    # Compute the third column of the transformation matrix M
    third_row = (im_pos_all[-1, 1:] - im_pos_M) / (im_pos_all.shape[0] - 1)
    transf_mat[:-1, 2] = third_row 
    
    voxel_coords = np.argwhere(np.ones(img_3d.shape))
    values = np.nan_to_num(img_3d).flatten()
    coords = voxel2space(transf_mat, voxel_coords)

    return coords, values


def load_data_nifti(filename: str):
    """
    Retrieves the surface points of the 3D volume defined by a NIfTI file.
    
    Parameters
    ----------
    filename : str 
        Path to the NIfTI file.
        
    Returns
    -------
    coords : Numpy array
        Contains the spatial coordinates of the z-stacked images.

    values : Numpy array
        Contains the parameter values for each of the points in coords array.
        
    """

    nifti_file = nib.load(filename)
    transf_mat = nifti_file.affine
    mask_data = np.array(nifti_file.dataobj)
    
    voxel_coords = np.argwhere(np.ones(mask_data.shape))
    values = np.nan_to_num(mask_data).flatten()
    coords = voxel2space(transf_mat, voxel_coords)

    return coords, values


def interpolate_data(coords, values, centroids,
                     output_dir: str,
                     interp_method: str = 'nearest'):
    """
    

    Parameters
    ----------
    coords : Numpy array
        Contains the spatial coordinates of the z-stacked images.
    values : Numpy array
        Contains the parameter values for each of the points in coords array.
    centroids : Numpy array 
        Centroids of the elements in the 3D mesh generated previously
    output_dir : string
        Destination folder
    method : str, optional
        Method used in the Scipy's griddata interpolation function. The default is 'nearest'.

    Returns
    -------
    None.

    """
    print("Interpolating values to the centroids...")
    centroid_values = griddata(coords, values, centroids, method=interp_method)
    print("Done")
    print("Saving interpolated values to a file...")
    np.savetxt(os.path.join(output_dir,'centroid_values.csv'), centroid_values, delimiter=",", fmt='%10.5f')
    print("Done")
    return
    
def main(im_data_path: str, input_format_data: str, output_dir: str, interp_method: str, centroids):
    if os.path.isfile(im_data_path):
        if input_format_data.lower() == 'nifti':
            print("Loading imaging data...")
            coords, values = load_data_nifti(im_data_path)
            print("Done")
            interpolate_data(coords, values, centroids, output_dir, interp_method)
        else:
            raise Exception(
                '{file} is not a valid format. When choosing a path to a file, only NIfTI format is supported.'.format(
                    file=im_data_path))
        
    elif os.path.isdir(im_data_path):
        if input_format_data.lower() == 'dicom':
            print("Loading imaging data...")
            coords, values = load_data_dicom(im_data_path)
            print("Done")
            interpolate_data(coords, values, centroids, output_dir, interp_method)
        else:
            raise Exception(
                '{file} is not a valid format. When choosing a path to a folder, only DICOM format is supported.'.format(
                    file=im_data_path))
    return

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Image data interpolation to FE mesh')
    parser.add_argument('-b', '--batch', action='store_true',
                        help=("Run the code in batch mode.\n"
                              "If selected, the user must set the input parmeters using arguments.\n"
                              "Type --help for more information."))
    parser.add_argument('--im_data_path', type=str,
                        help="Path to imaging data file or folder.")
    parser.add_argument('--input_format_data', type=str,
                        help="Image data file format. Recognized formats: DICOM and NIfTI.\n"
                              "If the input is a NIfTI file, --im_data_path must "
                              "be a file path. Else, it must be a folder path.")
    parser.add_argument('--output_dir', type=str, default=os.getcwd(),
                        help="Path to output folder.")
    parser.add_argument('--interp_method', type=str,
                        help="Interpolation method used in Scipy's griddata function.")
    parser.add_argument('--centroids_path', type=str,
                        help="Path to the file containing the elements centroids.")
    args = parser.parse_args()

    if args.batch:
        centroids = np.genfromtxt(args.centroids_path, delimiter=',')
        main(args.im_data_path, args.input_format_data, args.output_dir, args.interp_method, centroids)
    else:
        input_format_data = select_input_format_data()
        if input_format_data is None:
            sys.exit("Invalid input format")
        elif input_format_data.lower() == 'dicom':
            message_folder(input_format_data)
            im_data_path = diropenbox(msg="Choose a folder")
        else:
            im_data_path = fileopenbox(msg="Select the input file")
        if im_data_path is None:
            sys.exit("Invalid path")
        
        centroids_path = fileopenbox(msg="Select the file containing the centroids")
        centroids = np.genfromtxt(centroids_path, delimiter=',')
        message_out_dir(os.getcwd())
        output_dir = diropenbox(msg="Choose a destination folder")
        interp_method=interp_method_selection().lower()
        main(im_data_path=im_data_path,
             input_format_data=input_format_data,
             output_dir=output_dir,
             interp_method=interp_method,
             centroids=centroids
             )
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    