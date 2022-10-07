''' Code containing all GUI functions'''

from easygui import *

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

def select_input_format_data():
    text = ("Select the file format of the imaging data to interpolate:\n"
            "If you choose DICOM, please select the directory where"
            "the files are stored.\nIf you choose NIfTI, select the input file.")
    title = "File format selection"
    button_list = []
    button1 = "DICOM"
    button2 = "NIfTI"
    button3 = "CSV"
    button_list.append(button1)
    button_list.append(button2)
    button_list.append(button3)
    input_ft_data = buttonbox(text, title, button_list)
    return input_ft_data

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
    text = ("Do you want to export an array containing the element centroids to a csv file?\n"
            "Note that if the user later chooses to interpolate imaging data to the mesh, this option will be forced to True")

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

def interp_method_selection():
    text = ("Select theinterpolation method:\n"
            "Defaults to 'nearest'")
    title = "Interpolation method selection"
    button_list = []
    button1 = "Nearest"
    button2 = "Linear"
    button3 = "Cubic"
    button_list.append(button1)
    button_list.append(button2)
    button_list.append(button3)
    interp_method = buttonbox(text, title, button_list)
    return interp_method

def select_interpolate2mesh():
    text = "Do you want to interpolate imaging data to the FE mesh?"

    # window title
    title = "Interpolate to mesh"

    interpolate2mesh = boolbox(text, title, ('Yes', 'No'), default_choice='No')
    return interpolate2mesh