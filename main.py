# -*- coding: utf-8 -*-
"""
This library transforms a set of binarized z-slices into a 3D mesh.
These slices can be read from different formats:
- .dcm
- .dcmseg
- .nifti
- .jpg, .tif, .png, ... [any image format accepted by opencv]
@author: Diego Sainz (dsainz@unizar.es) & Carlos Borau (cborau@unizar.es)
"""

import numpy as np
import formatreader as fmt_r
import formatwriter as fmt_w
import visualization as vis
from easygui import *
import argparse
import sys
import os


def file_folder_selection():
    text = ("Select the file format of your segmentation:\n"
            "If you choose DICOM or IMAGE, please select the directory where"
            "the files are stored.\nIf you choose other formats, select the input file.")
    title = "File format selection"
    button_list = []
    button1 = "DICOM"
    button2 = "DICOM-SEG"
    button3 = "IMAGE"
    button4 = "NIfTI"
    button_list.append(button1)
    button_list.append(button2)
    button_list.append(button3)
    button_list.append(button4)
    input_ft = buttonbox(text, title, button_list)
    return input_ft


def message_folder(input_ft: str):
    message = ("You have selected " + input_ft + " as the input format.\n"
                                                     "Note that you must select the DIRECTORY where the files are stored.")
    title = "Information"
    ok_btn = "Continue"

    msgbox(message, title, ok_btn)


def select_smooth():
    text = ("Do you want to smooth the slices before interpolating them?\n"
            "It is highly recommended if your geometry contains sharp edges, small holes, etc.")

    # window title
    title = "Smooth segmentation"

    output_smooth = boolbox(text, title, ('Yes', 'No'), default_choice='Yes')
    return output_smooth


def set_n_interp():
    text = ("Number of interpolations: Number of interpolates slices created"
            " between two adjacent slices in the segmentation.\n"
            "Default is 10.")

    # window title
    title = "Set number of interpolations"
    n_interp = integerbox(text, title, default=10, lowerbound=0, upperbound=99999999)
    return n_interp


def set_target_faces():
    text = ("Approximate wished number of faces of the generated STL.\n"
            "Default is 50000.")

    # window title
    title = "Set target faces for the STL"
    target_faces = integerbox(text, title, default=50000, lowerbound=10, upperbound=99999999)
    return target_faces


def set_z_size():
    text = ("Distance in z-dir between images.\n"
            "Default is 30.")

    # window title
    title = "Set z-distance between consecutive images"
    z_size = enterbox(text, title, default="30.0")
    return float(z_size)


def set_mask_id():
    text = ("List of mask ids (delimited by commas), labeled in the file, to be merged.\n"
            "Default is [0], which means that all labels will be merged.")

    # window title
    title = "Set list of mask ids (delimited by commas)"
    mask_id_str = enterbox(text, title, default="0")
    mask_id = [int(item) for item in mask_id_str.split(',')]
    return mask_id


def set_boundary_weight():
    text = ("Boundary weight: Sets the importance of the boundary on the simplification algorithm.\n"
            "If it equals 1.0, it means that the boundary has the same importance of the rest.\n"
            "Default is 0.5.")

    # window title
    title = "Set boundary weight for STL recosntruction"
    boundary_weight = enterbox(text, title, default="0.5")
    return float(boundary_weight)
    
    
def set_sampling_factor():
    text = ("Reduction ratio applied to the initial pointcloud. Ranges from 0 to 1.\n"
            "Default is 0.1.")

    # window title
    title = "Set sampling factor for pointcloud simplification"
    sampling_factor = enterbox(text, title, default="0.1")
    return float(sampling_factor)

def select_laplacian():
    text = ("Do you want to perform smooth the reconstructed STL prior to generating the 3D mesh?\n"
            "The surface mesh will be smoothed using the Laplacian Smoothing Method")

    # window title
    title = "Laplacian Smoothing"

    return_laplacian = boolbox(text, title, ('Yes', 'No'), default_choice='Yes')
    return return_laplacian

def set_mesh_size():
    text = ("Min. mesh size: Minimum element size of the mesh created.\n"
            "Max. mesh size: Maximum element size of the mesh created.\n"
            "Both default to 2.5 mm")

    # window title
    title = "Set mesh size"
    min_mesh_values, max_mesh_values = enumerate(multenterbox(text, title, ['Minimim mesh size', 'Maximum mesh size']))
    #Set default values
    if len(min_mesh_values[1])==0:
        min_mesh_size=2.5
    elif len(min_mesh_values[1])>0:
        min_mesh_size = float(min_mesh_values[1])
    if len(max_mesh_values[1])==0:
        max_mesh_size = 2.5
    elif len(max_mesh_values[1])>0:
        max_mesh_size = float(max_mesh_values[1])
    return min_mesh_size, max_mesh_size


def message_out_dir(current_folder):
    message = ("Please select the destination folder. Default is current working directory, which is:\n" +
               current_folder)
    title = "Destination folder"
    ok_btn = "Continue"

    msgbox(message, title, ok_btn)


def set_output_prefix():
    text = ("Prefix for output files name.\n"
            "Default is 'output'.")

    # window title
    title = "Output files prefix"
    output_prefix = enterbox(text, title, default="output")
    return output_prefix


def select_centroids():
    text = ("Do you want to export an array containing the element centroids to a csv file?")

    # window title
    title = "Export centroids"

    return_centroids = boolbox(text, title, ('Yes', 'No'), default_choice='Yes')
    return return_centroids


def select_vtk():
    text = "Do you want to export the mesh generated to a vtk file?"

    # window title
    title = "Export vtk"

    export_vtk = boolbox(text, title, ('Yes', 'No'), default_choice='Yes')
    return export_vtk

def select_visualization():
    text = "Do you want to view the 3D representation of the external pointcloud prior to STL reconstruction?"

    # window title
    title = "View pointcloud"

    view_pcd = boolbox(text, title, ('Yes', 'No'), default_choice='No')
    return view_pcd

def mesh_format_selection():
    text = ("Select the mesh format:\n"
            "The algorithm generates by default a .dat mesh file."
            " Close this window if you do not want additional formats.")
    title = "Mesh format selection"
    button_list = []
    button1 = "Abaqus"
    button2 = "ANSYS"
    button_list.append(button1)
    button_list.append(button2)
    mesh_format = buttonbox(text, title, button_list)
    return mesh_format


def main(selected_path, input_format, n_interp=10, smooth_slices=False, z_size=30.0, mask_id=[0], target_faces=50000,
         boundary_weight=0.5, sampling_factor=0.1, laplacian_smoothing=True, min_mesh_size=2.5, max_mesh_size=2.5, output_dir: str = os.getcwd(),
         output_prefix: str = 'output',
         save_centroids=False, export_vtk=False, mesh_format=None, view_pcd=False):
    contours_3d, covers, transf_mat = fmt_r.get_mask(selected_path=selected_path,
                                                     file_format=input_format.lower(),
                                                     n_interp=n_interp,
                                                     z_size=z_size,
                                                     mask_id=mask_id,
                                                     smooth_slices=smooth_slices
                                                     )
    external_points = np.append(contours_3d, covers, axis=0)
    total_points_3d = fmt_r.voxel2space(transf_mat, external_points)
    if view_pcd:
        vis.show_point_cloud(total_points_3d)
    output_file_prefix = output_prefix
    ouput_path = os.path.join(output_dir, output_prefix)
    fmt_w.stl_from_3dply(total_points_3d,
                         output_dir,
                         output_prefix,
                         target_faces,
                         boundary_weight,
                         sampling_factor,
                         laplacian_smoothing
                         )
    nodes, elements = fmt_w.mesh3d_from_stl(output_dir,
                                            output_prefix,
                                            min_mesh_size=min_mesh_size,
                                            max_mesh_size=max_mesh_size,
                                            export_vtk=export_vtk,
                                            save_centroids=save_centroids
                                            )
    if mesh_format is not None:
        if mesh_format.lower() == 'ansys':
            fmt_w.write_ansys_inp(nodes,
                                  elements,
                                  output_dir,
                                  output_prefix)
        if mesh_format.lower() == 'abaqus':
            fmt_w.write_abaqus_inp(nodes,
                                   elements,
                                   output_dir,
                                   output_prefix)

    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='image2mesh')
    parser.add_argument('-b', '--batch', action='store_true',
                        help=("Run the code in batch mode.\n"
                              "If selected, the user must set the input parmeters using arguments.\n"
                              "Type --help for more information."))
    parser.add_argument('--path', type=str,
                        help="Path to input folder or file.")
    parser.add_argument('--format', type=str,
                        help=("Input file format. Recognized formats: DICOM, DICOM-SEG, NIfTI, IMAGE.\n"
                              "If the input is a DICOM-SEG or NIfTI file, --path must "
                              "be a file path. Else, it must be a folder path."))
    parser.add_argument('--n_interp', type=int, default=10,
                        help=("Number of interpolates slices created"
                              " between two adjacent slices in the segmentation.\n"
                              "Default is 10"))
    parser.add_argument('--z_size', type=float, default=30.0,
                        help=("Distance between consecutive slices.\n"
                              "Default is 1.0"))
    parser.add_argument('--mask_id', help='Input mask ids delimited by commas. Only necessary for DICOM-SEG and NIfTI inputs.',
                        type=lambda s: [int(item) for item in s.split(',')])
    parser.add_argument('--smooth_slices', action='store_true',
                        help="If True, the slices will be smooothed before interpolating them.")
    parser.add_argument('--target_faces', type=int, default=50000,
                        help=("Number of faces of the STL generated.\n"
                              "Default is 50000."))
    parser.add_argument('--boundary_weight', type=float, default=0.5,
                        help=("Sets the importance of the boundary on the simplification algorithm.\n"
                              "If it equals 1.0, it means that the boundary has the same importance of the rest.\n"
                              "Default is 0.5."))
    parser.add_argument('--sampling_factor', type=float, default=0.1,
                        help=("Reduction ratio applied to the initial pointcloud.\n"
                              "Default is 0.1."))
    parser.add_argument('--laplacian_smoothing', action='store_true',
                        help="If True the algorithm will smooth the STL following the Laplacian Smoothing Method.")
    parser.add_argument('--min_mesh_size', type=float, default=2.5,
                        help="Minimum element size of the mesh created.\nDefault is 2.5 mm")
    parser.add_argument('--max_mesh_size', type=float, default=2.5,
                        help="Maximum element size of the mesh created.\nDefault is 2.5 mm")
    parser.add_argument('--output_dir', type=str, default=os.getcwd(),
                        help="Path to output folder.")
    parser.add_argument('--output_prefix', type=str, default='output',
                        help="Prefix for output files names")
    parser.add_argument('--save_centroids', action='store_true',
                        help="If True the algorithm will create a .csv file storing the elements centroids.")
    parser.add_argument('--export_vtk', action='store_true',
                        help="If True, the obtained mesh will be exported as a .vtk file.")
    parser.add_argument('--mesh_format', type=str, default=None,
                        help="Choose the mesh format: Abaqus or Ansys.")
    parser.add_argument('--view_pcd', action='store_true',
                        help="If True an additional window will show the 3D visualization of the pointcloud before reconstructing the STL")

    args = parser.parse_args()

    if args.batch:
        main(args.path, args.format, args.n_interp, args.smooth_slices,
             args.z_size, args.mask_id,
             args.target_faces, args.boundary_weight, args.sampling_factor, args.laplacian_smoothing,
             args.min_mesh_size, args.max_mesh_size, args.output_dir, args.output_prefix, args.save_centroids, 
             args.export_vtk, args.mesh_format)
    else:
        input_format = file_folder_selection()
        if input_format is None:
            sys.exit("Invalid input format")
        elif input_format.lower() == 'dicom' or input_format.lower() == 'image':
            message_folder(input_format)
            selected_path = diropenbox(msg="Choose a folder")
        else:
            selected_path = fileopenbox(msg="Select the input file")
        if selected_path is None:
            sys.exit("Invalid path")
        smooth_slices = select_smooth()
        n_interp = set_n_interp()
        if input_format.lower() == 'image':
            z_size = set_z_size()
        else:
            z_size = 1.0
        if input_format.lower() == 'dicom-seg' or input_format.lower() == 'nifti':
            mask_id = set_mask_id()
        else:
            mask_id = [0]

        target_faces = set_target_faces()
        boundary_weight = set_boundary_weight()
        sampling_factor = set_sampling_factor()
        laplacian_smoothing = select_laplacian()
        min_mesh_size, max_mesh_size = set_mesh_size()
        message_out_dir(os.getcwd())
        output_dir = diropenbox(msg="Choose a destination folder")
        output_prefix = set_output_prefix()
        save_centroids = select_centroids()
        export_vtk = select_vtk()
        mesh_format = mesh_format_selection()
        view_pcd = select_visualization()
        main(selected_path=selected_path,
             input_format=input_format,
             n_interp=n_interp,
             z_size=z_size,
             mask_id=mask_id,
             smooth_slices=smooth_slices,
             target_faces=target_faces,
             boundary_weight=boundary_weight,
             sampling_factor=sampling_factor,
             laplacian_smoothing=laplacian_smoothing,
             min_mesh_size=min_mesh_size,
             max_mesh_size=max_mesh_size,
             output_dir=output_dir,
             output_prefix=output_prefix,
             save_centroids=save_centroids,
             export_vtk=export_vtk,
             mesh_format=mesh_format,
             view_pcd=view_pcd)
