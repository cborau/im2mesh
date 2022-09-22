''' Code containing all functions related to format reading and mask obtention'''
import numpy as np
import sys
import re
import sliceinterpolator as slici
import cv2
import pydicom_seg
import os
import pydicom as pyd
from pathlib import Path
from skimage.measure import label
import visualization as vis
import nibabel as nib


def get_largest_CC(mask):
    """
        Retrieves the largest connected component in a 3D volume (biggest separated object).

        Parameters
        ----------
        mask : Numpy array
            A 3D binarized volume.

        Returns
        -------
        mask : Numpy array (float)
            A 3D binarized volume containing only the largest object.

    """
    mask_bin=(mask > 0).astype(np.uint8) #Make sure mask is binary
    labels = label(mask_bin, connectivity=1)
    if labels.max() != 0:
        largest_cc = labels == np.argmax(
            np.bincount(labels.flat)[1:]) + 1  # +1 because 0, which is background, is eliminated
    else:
        print('ERROR: mask volume is empty')
        sys.exit(1)
    return largest_cc.astype(float)


def get_mask(selected_path: str, file_format: str, n_interp: int = 10, smooth_slices: bool = True, **kwargs):
    """
    Calls the corresponding function depending on the file format selected and retrieves its outputs.

    Parameters
    ----------
    selected_path : string 
        Path to the selected folder or file.
    
    file_format : string
        File format chosen in the GUI (nifti, tiff, dicom-seg, image).
    
    n_interp : int
        Number of interpolated slices to generate between two real slices. The default is 10.

    smooth_slices : boolean
        If True, calls the function smooth_mask() on each of the real slices. The default is True.

    **kwargs
        Additional keyword arguments passed.
        mask_id : list of int
            Merge the specified mask ids into a single volume.
            Default = [0]. If [0] all masks are merged into a single point cloud.
        zsize : float
            Distance between slices. Needed when loading slices from multiple single images.

    Returns
    -------
    contours : Numpy array 
        Contains the contours of every slice in voxel coordinates.

    covers : Numpy array
        Contains the points from the top and bottom slices. Needed to close the .stl geometry.
        
    transf_mat : Numpy array
        Transformation matrix to convert from voxel coordinates to space coordinates.

    """
    mask_id = kwargs.get('mask_id', [0])
    z_size = kwargs.get('z_size', 30.0)

    if os.path.isfile(selected_path):
        ext = os.path.splitext(selected_path)[-1].lower()
        if (ext == '.dcm') or (ext == '.seg') and (file_format == 'dicom-seg'):
            contours, covers, transf_mat = get_mask_dcmseg(selected_path, mask_id=mask_id, n_interp=n_interp,
                                                           smooth_slices=smooth_slices)
        elif ext == '.nii' and (file_format == 'nifti'):
            contours, covers, transf_mat = get_mask_nifti(selected_path, mask_id, n_interp=n_interp, smooth_slices=smooth_slices)
        elif ((ext == '.tif') or (ext == '.tiff') or (ext == '.jpg') or (ext == '.png')) and (file_format == 'image'):
            # If user selects a single image, use the parent folder
            selected_path = os.path.dirname(os.path.abspath(selected_path))
            contours, covers, transf_mat = get_mask_from_images(selected_path, ext, z_size=z_size, n_interp=n_interp,
                                                                smooth_slices=smooth_slices)
        else:
            print(
                '{file} is not a valid format. When choosing a path to a file, only DICOM and NIfTI formats are supported.'.format(
                    file=selected_path))
    elif os.path.isdir(selected_path):
        if file_format == 'dicom':
            for file in os.listdir(selected_path):
                ext = os.path.splitext(file)[-1].lower()
                if (ext == '.dcm'):
                    pass
                else:
                    raise Exception(
                        "Incorrect file extension in file {file}. Inputs must be .dcm format.".format(
                            file=file))
            contours, covers, transf_mat = get_mask_dicom(selected_path, n_interp=n_interp, smooth_slices=smooth_slices)
        elif file_format == 'image':
            count = 0
            for file in os.listdir(selected_path):
                ext = os.path.splitext(file)[-1].lower()
                if (ext == '.tif') or (ext == '.tiff') or (ext == '.jpg') or (ext == '.png'):
                    count += 1
            if count < 2:
                raise Exception(
                    "At least 2 files of {file} should be in the folder".format(
                        file=file))
            print('Input files in folder checked for correct extension. Starting the algorithm ...')
            contours, covers, transf_mat = get_mask_from_images(selected_path, ext, z_size=z_size, n_interp=n_interp,
                                                                smooth_slices=smooth_slices)

    else:
        print('Not such file or directory: {file}'.format(file=selected_path))
    return contours, covers, transf_mat


def get_numbers_from_filename(filename):
    """
    Read numbers in a filename. If there are various numbers, only returns the first occurrence.
    E.g. in 'this1is25a_test001' the function only returns '1'.

    Parameters
    ----------
    filename : string
        Name of the file

    Returns
    -------
    s : string
        Returns the number found.

    """
    return re.search(r'\d+', filename).group(0)


def smooth_mask(img, n_iter=1, circle_size=5):
    """
    Smooths the input image via erosion/dilation operations.

    Parameters
    ----------
    img : Numpy array
        A 2D binarized image.

    n_iter : int
        Number of erosion/dilation operations to be performed.

    circle_size : int
        Size of the structuring element to perform the erosion/dilation operations.


    Returns
    -------
    Numpy array
        The smoothed version of the input image.

    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (circle_size, circle_size))
    image_er = cv2.erode(img, kernel, iterations=n_iter)
    image_er_dil = cv2.dilate(image_er, kernel, iterations=n_iter)
    # image_dil = cv2.dilate(img, kernel, iterations=n_iter)
    # image_dil_er = cv2.erode(image_dil, kernel, iterations=n_iter)
    if image_er_dil.max() == 0.0:
        print('Structuring element size ({0} pixels) was too big and eroded the entire slice'.format(circle_size))
        print('Returning original slice instead')
        image_er_dil = img
    return image_er_dil


def dilate_countour(img, n_iter=1, circle_size=3):
    """
    Dilates the image to capture the external contour afterwards.

    Parameters
    ----------
    img : Numpy array
        A 2D binarized image.

    n_iter : int
        Number of erosion/dilation operations to be performed.

    circle_size : int
        Size of the structuring element to perform the dilation operation.


    Returns
    -------
    Numpy array
        The smoothed version of the input image.

    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (circle_size, circle_size))
    image_dil = cv2.dilate(img, kernel, iterations=n_iter)

    return image_dil


def get_mask_from_images(dirname, ext, z_size=30.0, n_interp=10, smooth_slices=True):
    """
    Retrieves the surface points of the 3D volume defined in multiple individual images.
    Images are automatically binarized via Otsu thresholding.

    Parameters
    ----------
    dirname : string
        The directory containing the images.

    ext : string
        File extension (ej: .jpg, .png).

    z_size : float
        Vertical distance between slices.

    n_interp : int
        Number of slices interpolated between every consecutive original images. Default = 10.

    smooth_slices : bool
        If True, original binarized images are smoothed via erosion/dilation operations. Default = True


    Returns
    -------
    contours : Numpy array
        Contains the contours of every slice in voxel coordinates.

    covers : Numpy array
        Contains the points from the top and bottom slices. Needed to close the .stl geometry.

    transf_mat : Numpy array
        Transformation matrix to convert from voxel coordinates to space coordinates.

    """

    count = 0
    for file in os.listdir(dirname):
        if file.endswith(ext):
            count += 1
            fullname = os.path.join(dirname, file)
            # binarize image and change type to float64 for interpolation
            image = cv2.imread(fullname, cv2.IMREAD_GRAYSCALE)
            th, bw = cv2.threshold(image, 128, 255, cv2.THRESH_OTSU)
            bw = bw.astype('float64')
            bw = bw / 255.0
            if count == 1:  # first image
                mask = bw
            else:
                mask = np.dstack((mask, bw))
            print('-loading: ' + os.path.join(dirname, file))

    # Check for multiple connected components and keep the biggest one
    mask = get_largest_CC(mask)
    # Get only the slices where there is part of the segmentation
    slices = []
    for i in range(mask.shape[2]):
        if np.argwhere(mask[:,:,i]).astype(np.float32).size != 0:
            slices.append(i)
    contours_3d = np.array([])
    covers = np.array([])
    # smooth slices
    if smooth_slices:
        for i in range(len(slices)):
            mask[:, :, slices[i]] = smooth_mask(mask[:, :, slices[i]], n_iter=1)

    contours_3d, covers = contour_from_3dmask(mask, slices, n_interp)

    transf_mat = np.eye(4)
    transf_mat[3, :] = 0
    third_row = np.array([0,0,(np.max(slices) - np.min(slices)) / (len(slices) - 1)])
    transf_mat[:-1, 2] = third_row * z_size
    transf_mat[:, 3] = 1
    return contours_3d, covers, transf_mat


def get_mask_dcmseg(filename: str, mask_id=[0], n_interp=10, smooth_slices=True):
    """
    Retrieves the surface points of the 3D volume defined by the masks of a DICOM seg file.

    Parameters
    ----------
    filename : str must be .dcm format
    
    mask_id : int default = 0. If 0 all masks are merged into a single point cloud

    n_interp : int
        Number of slices interpolated between every consecutive original slices. Default = 10.

    smooth_slices : bool
        If True, binary slices are smoothed via erosion/dilation operations. Default = True


    Returns
    -------
    contours : Numpy array
        Contains the contours of every slice in voxel coordinates.

    covers : Numpy array
        Contains the points from the top and bottom slices. Needed to close the .stl geometry.

    transf_mat : Numpy array
        Transformation matrix to convert from voxel coordinates to space coordinates.

    """

    dcm = pyd.dcmread(filename)
    reader = pydicom_seg.MultiClassReader()
    result = reader.read(dcm)
    voxel_size_x, voxel_size_y, voxel_size_z = result.spacing
    image_data = result.data  # directly available

    if (len(mask_id) == 1) and (mask_id[0] != 0):
        mask = (image_data == mask_id[0]).astype(np.float64)
    else:
        if (len(mask_id) == 1) and (mask_id[0] == 0):  # mask_id = [0] means get all masks
            mask_ids = np.unique(image_data)
            mask_ids = mask_ids[mask_ids > 0]
        else:
            mask_ids = mask_id
        mask = np.zeros(image_data.shape)

        for m in mask_ids:
            mask_m = image_data == m
            mask += mask_m

    mask = np.moveaxis(mask, 0, -1)  # swap axes so they are in y,x,z order
    mask = np.moveaxis(mask,0,1) # swap axes so they are in x,y,z order
    # Check for multiple connected components and keep the biggest one
    mask_label = get_largest_CC(mask)
    # Get only the slices where there is part of the segmentation
    slices = []
    for i in range(mask_label.shape[2]):
        if np.argwhere(mask_label[:,:,i]).astype(np.float32).size != 0:
            slices.append(i)
    # smooth slices
    if smooth_slices:
        for i in range(len(slices)):
            mask_label[:, :, slices[i]] = smooth_mask(mask_label[:, :, slices[i]], n_iter=1)

    contours_3d, covers = contour_from_3dmask(mask_label, slices, n_interp)

    M1 = result.direction
    T1 = np.array(result.origin).reshape(-1, 1)
    sp = np.array(result.spacing).reshape(-1, 1)
    add_row = np.array([0, 0, 0, 1]).reshape(-1, 1).T
    
    sp_matrix = np.array([[sp[0,0],0,0],[0,sp[1,0],0],[0,0,sp[2,0]]])
    rotation_scaling = np.matmul(M1,sp_matrix)

    transf_mat = np.append(np.append(rotation_scaling, T1, axis=1), add_row, axis=0)
    return contours_3d, covers, transf_mat


def get_mask_dicom(directory, n_interp: int = 10, smooth_slices: bool = True):
    """
    Retrieves the surface points of the 3D volume defined by a DICOM file.
    
    Parameters
    ----------
    directory : str 
        Path to the folder containing the DICOM files.
    
    n_interp : int
        Number of slices interpolated between every consecutive original slices. Default = 10.

    smooth_slices : boolean
        If True, binary slices are smoothed via erosion/dilation operations. Default = True


    Returns
    -------
    contours : Numpy array
        Contains the contours of every slice in voxel coordinates.

    covers : Numpy array
        Contains the points from the top and bottom slices. Needed to close the .stl geometry.

    transf_mat : Numpy array
        Transformation matrix to convert from voxel coordinates to space coordinates.
        
    """

    im_pos_all = np.array([])
    #Get the first dicom in the dir to get the image size
    init_file = os.listdir(directory)[0] 
    n_files = len(os.listdir(directory))
    ds_init = pyd.dcmread(Path(directory, init_file))
    img_init = ds_init.pixel_array
    mask = np.empty((img_init.shape[0],img_init.shape[1], n_files)) #Initialize empty mask array with initial image size
    first_slice = True
    for file in os.listdir(directory):
        if file.endswith(".dcm"):
            ds = pyd.dcmread(Path(directory, file))
            img = ds.pixel_array
            slice_number = int(get_numbers_from_filename(file))
            mask[:,:,slice_number] = img
            # Extract the necessary parameters from DICOM header
            im_pos = list(map(float, ds.ImagePositionPatient))  # Image position
            im_pos_number = np.array([slice_number] + im_pos)
            im_pos_all = np.append(im_pos_all, im_pos_number).reshape(-1, 4)
            if first_slice:
                im_pos_M = np.array(
                    ds.ImagePositionPatient)  # Image position of the first slice to compute the transformation matrix M
                im_or = np.array(ds.ImageOrientationPatient)  # Image orientation
                # slice_thickness = float(ds.SliceThickness)  # Slice thickness
                pix_sp = np.array(ds.PixelSpacing)  # Pixel spacing
                first_slice = False
                # Transformation matrix
            # We first compute the transformation matrix for a single slice
            # Link to source: https://nipy.org/nibabel/dicom/dicom_orientation.html#dicom-voxel-to-patient-coordinate-system-mapping
            # Image orientation values are flipped (columnwise) due to the DICOM coordinate system
    transf_mat = np.array([[im_or[3] * pix_sp[0], im_or[0] * pix_sp[1], 0, im_pos_M[0]],
                           [im_or[4] * pix_sp[0], im_or[1] * pix_sp[1], 0, im_pos_M[1]],
                           [im_or[5] * pix_sp[0], im_or[2] * pix_sp[1], 0, im_pos_M[2]], [0, 0, 0, 1]])
    # Check for multiple connected components and keep the biggest one
    mask = get_largest_CC(mask)
    # Get only the slices where there is part of the segmentation
    slices = []
    for i in range(mask.shape[2]):
        if np.argwhere(mask[:,:,i]).astype(np.float32).size != 0:
            slices.append(i)
    # Compute the third column of the transformation matrix M
    third_row = (im_pos_all[-1, 1:] - im_pos_M) / (im_pos_all.shape[0] - 1)
    transf_mat[:-1, 2] = third_row 

    # smooth slices
    if smooth_slices:
        for i in range(len(slices)):
            mask[:, :, slices[i]] = smooth_mask(mask[:, :, slices[i]], n_iter=1)

    contours_3d, covers = contour_from_3dmask(mask, slices, n_interp)

    return contours_3d, covers, transf_mat


def get_mask_nifti(filename: str, mask_id=[0], n_interp: int = 10, smooth_slices: bool = True):
    """
    Retrieves the surface points of the 3D volume defined by a NIfTI file.
    
    Parameters
    ----------
    filename : str 
        Path to the NIfTI file.
        
    mask_id : int default = 0. If 0 all masks are merged into a single point cloud
    
    n_interp : int
        Number of slices interpolated between every consecutive original slices. Default = 10.

    smooth_slices : boolean
        If True, binary slices are smoothed via erosion/dilation operations. Default = True
    
    Returns
    -------
    contours_3d : Numpy array 
        Contains the contours of every slice in voxel coordinates.

    covers : Numpy array
        Contains the points from the top and bottom slices.
        
    transf_mat : Numpy array
        Transformation matrix to convert from voxel coordinates to space coordinates.
        
    """

    nifti_file = nib.load(filename)
    transf_mat = nifti_file.affine
    mask_data = np.array(nifti_file.dataobj)
    
    
    if (len(mask_id) == 1) and (mask_id[0] != 0):
        mask = ((np.rint(mask_data)).astype(int) == mask_id[0]).astype(np.float64)
    else:
        if (len(mask_id) == 1) and (mask_id[0] == 0):  # mask_id = [0] means get all masks
            mask_ids = np.unique(mask_data)
            mask_ids = mask_ids[mask_ids > 0]
        else:
            mask_ids = mask_id
        mask = np.zeros(mask_data.shape)
        for m in mask_ids:
            mask_m = mask_data == m
            mask += mask_m
            mask_m_coords = np.argwhere(mask_m).astype(np.float32)
    
    # Check for multiple connected components and keep the biggest one
    mask_label = get_largest_CC(mask)
    # Find the slices that contain the segmentation
    slices = np.unique(np.nonzero(mask_label)[-1])

    # dilate contour
    for i in range(len(slices)):
        mask_label[:, :, slices[i]] = dilate_countour(mask_label[:, :, slices[i]], n_iter=1)
    # smooth slices
    if smooth_slices:
        for i in range(len(slices)):
            mask_label[:, :, slices[i]] = smooth_mask(mask_label[:, :, slices[i]], n_iter=1)
    contours_3d, covers = contour_from_3dmask(mask_label, slices, n_interp)

    return contours_3d, covers, transf_mat


def contour_from_3dmask(mask, slices, n_interp: int = 10):
    """
    From a 3D array containing the semgentation and the list of slices, 
    it creates a set of interpolated slices between each pair of adjacent 
    real slices using the sliceinterpolator module. Finally, calls the function
    analyze_slice_opencv() on each slice (real and interpolated) to get its 
    contour and outpouts the external points of the whole segmentation and the covers.
    
    Parameters
    ----------
    mask : Numpy array
        Array containing the segmentation.
    slices : List
        List of the slices present in the array.
    n_interp :  int 
        Number of interpolated slices to generate between two real slices. The default is 10.

    Returns
    -------
    contours_3d : Numpy array 
        Contains the contours of every slice in voxel coordinates.

    covers : Numpy array
        Contains the points from the top and bottom slices.

    """

    covers = np.array([])
    contours_3d = np.array([])
    prefix = 'Processing ' + str(len(slices)) + ' slices:'
    vis.progressbar(0, len(slices), prefix=prefix, suffix='complete', length=10)
    for i in range(len(slices)):
        vis.progressbar(i + 1, len(slices), prefix=prefix, suffix='complete', length=10)
        if (slices[i] == np.min(slices)) or (slices[i] == np.max(slices)):
            s_coords = np.argwhere(mask[:, :, slices[i]]).astype(np.float32)
            cover = np.append(s_coords, np.ones((len(s_coords), 1)) * slices[i], axis=1).reshape(
                -1, 3)
            covers = np.append(covers, cover).reshape(-1, 3)
        if i < len(slices) - 1:
            for p in range(0, n_interp + 1):
                interp_slices = slici.interpshape(mask[:, :, slices[i]], mask[:, :, slices[i + 1]], p / (n_interp)) * 1
                z_idx = (slices[i + 1] - slices[i]) * (p / n_interp) + slices[i]
                contours_3d = np.append(contours_3d,
                                        analyze_slice_opencv(interp_slices, z_pos=z_idx)).reshape(-1, 3)

    return contours_3d, covers


def analyze_slice_opencv(mask, z_pos: float, show_plots=False):
    """
    OpenCV implementation to obtain the contours present in the slice.
    
     Parameters
     ----------
     mask : Numpy array
         Binary array of the selected slice.
     
     z_pos : float
         Z coordinate of the selected slice.
     
     show_plots : boolean default = False
         If true, shows the contour extraction for the selected slice. 
         Press any key to close the visualization window.
     
    
     Returns
     -------
     contours_zidx : Numpy array
         Returns the coordinates of the points of the contour (X,Y,Z).

    """

    img = np.array(mask * 255, dtype=np.uint8)  # transforms Trues to 255 values
    contours, hier = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        contours_zidx = np.array([])
        if show_plots:
            image_color = np.zeros([img.shape[0], img.shape[0], 3])
            image_color[:, :, 0] = img * 64 / 255
            image_color[:, :, 1] = img * 128 / 255
            image_color[:, :, 2] = img * 192 / 255
            cv2.drawContours(image=image_color, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2,
                             lineType=cv2.LINE_AA)
            cv2.imshow('window', image_color)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        for i in range(len(contours)):
            contours_zidx = np.append(contours_zidx,
                                      np.append(np.fliplr(contours[i].reshape(-1, 2)),
                                                np.ones((len(contours[i]), 1)) * z_pos,
                                                axis=1)).reshape(-1, 3)
    else:
        pass

    return contours_zidx


def voxel2space(transformation_matrix, voxel_coordinates):
    """

    Transforms the coordinates in voxel CSYS to real world, using the transformation
    matrix generated using the get_val_pos() function.

    Parameters
    ----------
    transformation_matrix : Numpy array
        Transformation matrix

    voxel_coordinates : Numpy array
        Coordinates of the points in voxel CSYS

    Returns
    -------
    space_coordinates : Numpy array
        Coordinates of the points in real world CSYS

    """
    space_coordinates = transformation_matrix.dot(
        (np.append(voxel_coordinates, np.ones((len(voxel_coordinates), 1)), axis=1)).T).T[:, :3]
    return space_coordinates
