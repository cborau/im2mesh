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
import argparse
import sys
import os
from gui_options import *
from map_interpolation import main as main_interp


def main(selected_path, input_format, input_format_data, im_data_path: str, n_interp=10, smooth_slices=False, z_size=30.0, mask_id=[0], target_faces=50000,
         boundary_weight=0.5, sampling_factor=0.1, laplacian_smoothing=True, min_mesh_size=2.5, max_mesh_size=2.5, output_dir: str = os.getcwd(),
         output_prefix: str = 'output',
         save_centroids=False, export_vtk=False, mesh_format=None, view_pcd=False,
         interpolate2mesh=False, interp_method: str = 'nearest'):
    
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
    nodes, elements, centroids = fmt_w.mesh3d_from_stl(output_dir,
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
    if interpolate2mesh == True:
        input_format_data=input_format_data.lower()
        interp_method = interp_method.lower()
        main_interp(im_data_path,
                    input_format_data,
                    output_dir,
                    interp_method, 
                    centroids)

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
                        help="If True the algorithm will create a .csv file storing the elements centroids.\n"
                        "Note that if the user later chooses to interpolate imaging data to the mesh, this option will be forced to True")
    parser.add_argument('--export_vtk', action='store_true',
                        help="If True, the obtained mesh will be exported as a .vtk file.")
    parser.add_argument('--mesh_format', type=str, default=None,
                        help="Choose the mesh format: Abaqus or Ansys.")
    parser.add_argument('--view_pcd', action='store_true',
                        help="If True, an additional window will show the 3D visualization of the pointcloud before reconstructing the STL")
    parser.add_argument('--interpolate2mesh', action='store_true',
                        help="Whether there is data to interpolate to the FE mesh.\n"
                        "If true, the user will be asked to provide the path to the additional imaging data")
    parser.add_argument('--interp_method', type=str,
                        help="Interpolation method used in Scipy's griddata function.\n"
                        "Only used if intrepolate2mes is set to True")
    parser.add_argument('--input_format_data', type=str,
                        help="Image data file format. Recognized formats: DICOM and NIfTI.\n"
                              "If the input is a NIfTI file, --im_data_path must "
                              "be a file path. Else, it must be a folder path.\n"
                              "Additionally, a CSV containing a list of coordinates and values can be chosen as the input.")
    parser.add_argument('--im_data_path', type=str,
                        help="Path to imaging data file or folder.")
    args = parser.parse_args()

    if args.batch:
        if args.interpolate2mesh:
            args.save_centroids = True
        main(selected_path=args.path, input_format=args.format, n_interp=args.n_interp, smooth_slices=args.smooth_slices,
             z_size=args.z_size, mask_id=args.mask_id,
             target_faces=args.target_faces, boundary_weight=args.boundary_weight, sampling_factor=args.sampling_factor, 
             laplacian_smoothing=args.laplacian_smoothing, min_mesh_size=args.min_mesh_size, max_mesh_size=args.max_mesh_size, 
             output_dir=args.output_dir, output_prefix=args.output_prefix, save_centroids=args.save_centroids,
             export_vtk=args.export_vtk, mesh_format=args.mesh_format, interpolate2mesh=args.interpolate2mesh, 
             interp_method=args.interp_method, im_data_path=args.im_data_path, input_format_data=args.input_format_data)
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
        interpolate2mesh = select_interpolate2mesh()
        if interpolate2mesh:
            interp_method = interp_method_selection()
            input_format_data = select_input_format_data()
            im_data_path = fileopenbox(msg="Select the imaging data file")
        else:
            interp_method = None
            im_data_path = None
            
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
             view_pcd=view_pcd,
             interpolate2mesh = interpolate2mesh,
             interp_method = interp_method,
             im_data_path = im_data_path,
             input_format_data=input_format_data)
