''' Code containing all functions related to format writing (stl, gsmh mesh, ansys)'''

import gmsh
import pymeshlab
import numpy as np
import open3d as o3d
import visualization as vis
import os


def stl_from_3dply(total_points_3d,
                   output_dir: str,
                   output_prefix: str, 
                   target_faces: int = 5000,
                   boundary_weight: float = 0.5,
                   sampling_factor: float = 0.1):
    """
    
    Creates a 3D STL object from a given 3D point cloud using the pymeshlab library.
    It generates an intermediate .ply file where the point cloud is saved(generated for debugging purposes).
    
    Parameters
    ----------
    total_points_3d : numpy array
        Pointcloud (Nx3 dimensions)
    
    target_faces : int
        Total number of faces of the final STL (approx)
    
    boundary_weight : float
        Boundary Preserving Weight: The importance of the boundary during simplification.
        Default (1.0) means that the boundary has the same importance of the rest.
        Values greater than 1.0 raise boundary importance and has the effect of removing
        less vertices on the border. Admitted range of values (0,+inf).

    sampling_factor : float
        It is a ratio from 0 to 1 that sets the percentage of reduction in the number of points
        of the simplified pointcloud. 
        Reduced number of points = sampling_factor * initial number of points
        Defaults to 0.1. Low values are recommended for the robustness of the algorithm.
    
    filename_output : string
        Name of the output file containing the STL object

    """
    # target_faces is the total number of faces of the final STL (approx)
    # boundary_weight sets the importance of the boundary on the simplification algorithm. The higher, the more unaffected the boundary is.

    # First we save the pointcloud in a file format which is compatible with meshlab
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(total_points_3d)
    o3d.io.write_point_cloud(os.path.join(output_dir, output_prefix+".ply"), pcd)
    # Then we initiate meshlab and load this file
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(os.path.join(output_dir, output_prefix+".ply"))  # Load the ply mesh to meshlab
    n_samples = int(len(total_points_3d)*sampling_factor) # Number of samples of the simplifed pointcloud
    ms.apply_filter("generate_simplified_point_cloud", samplenum = n_samples)
    ms.apply_filter(
        'compute_normal_for_point_clouds')  # Apply filter to compute the normals from the given set of points
    ms.apply_filter('generate_surface_reconstruction_screened_poisson',
                    depth=8)  # Apply filter to reconstruct the surface based on those normals and the set of points
    ms.apply_filter('meshing_remove_duplicate_vertices')
    ms.apply_filter('apply_coord_hc_laplacian_smoothing')  # Smooth the surface obtained
    # ms.save_current_mesh('step1.stl') # Save the obatined surface as an STL file
    ms.apply_filter('meshing_decimation_quadric_edge_collapse', targetfacenum=target_faces, planarquadric=True,
                    autoclean=True, qualitythr=0.9, planarweight=0.1,
                    boundaryweight=boundary_weight)  # Apply filter to decimate the stl geometry
    # # https://pymeshlab.readthedocs.io/en/0.1.8/filter_list.html
    # #Targetfacenum is the number of faces we want the final geometry to have. Change this number if ANSYS meshing fails
    # #Planarquadric keeps triangles in the flat surfaces so the external mesh is not distorted. This must be always True
    # #planarweight is related to the previous feature. 0.1 seems to be the sweet spot for the cases avalible currently
    ms.apply_filter('meshing_remove_duplicate_vertices')
    ms.apply_filter('meshing_remove_duplicate_faces')
    ms.apply_filter('apply_coord_hc_laplacian_smoothing')  # Smooth the simplified surface
    ms.save_current_mesh(os.path.join(output_dir, output_prefix + ".stl"))  # Save the obatined surface as an STL file
    os.remove(os.path.join(output_dir, output_prefix+".ply"))
    return


def mesh3d_from_stl(output_dir: str,
                    output_prefix: str,
                    max_mesh_size: float = 5.0,
                    min_mesh_size: float = 2.5,
                    export_vtk: bool = True,
                    save_centroids: bool = False):
    """
    
    Generates a 3D FE mesh using linear tetrahedrons from a 3D .stl file using the
    GMSH library. It is saved in a .dat file.
    Returns two arrays containing node coordinates and element connectivity, respectively.
    
    Parameters
    ----------
    max_mesh_size, min_mesh_size : float
        Maximum and minimum size for the elements
    
    output_prefix : string
        Prefix for output files
        
    output_dir : string
        Destination folder
        
    export_vtk : boolean
        If True exports the mesh as a .vtk file
        
    save_centroids : boolean
        If True saves the element centroids coordinates as a .csv file.
    """
    gmsh.initialize()
    gmsh.merge(os.path.join(output_dir, output_prefix + ".stl"))
    gmsh.model.mesh.classifySurfaces(gmsh.pi, True, True, gmsh.pi)
    gmsh.model.mesh.createGeometry()
    s = gmsh.model.getEntities(2)
    surf = gmsh.model.geo.addSurfaceLoop([e[1] for e in s])
    gmsh.model.geo.addVolume([surf]) #create volume from surface
    gmsh.model.geo.synchronize()
    gmsh.option.setNumber('Mesh.Algorithm', 1)
    gmsh.option.setNumber('Mesh.MeshSizeMax', max_mesh_size)
    gmsh.option.setNumber('Mesh.MeshSizeMin', min_mesh_size)
    gmsh.model.mesh.generate(3) #set dimensions for mesh: 3D
    element_types, element_tags, node_tags = gmsh.model.mesh.getElements(dim=3)
    element_tags[0][:] = (element_tags[0][:] - element_tags[0][0]) + 1
    assert len(element_types) == 1 #make sure there is at least 1 3D element type
    name, dim, order, num_nodes, local_coords, num_first_order_nodes = \
    gmsh.model.mesh.getElementProperties(element_types[0]) #get element properties of the 3D element type
    connectivity = node_tags[0].reshape(-1, num_nodes)
    elems_connectivity = np.append(element_tags[0].reshape(-1,1),connectivity,axis=1) #element number + connectivity
    nodes = gmsh.model.mesh.getNodesByElementType(element_types[0])
    nodes_coords = nodes[1].reshape(-1,3)
    num_node_coords = np.append(nodes[0].reshape(-1,1),nodes_coords,axis=1)
    unique_nodes_coords = np.unique(num_node_coords,axis=0) #node number + coordinates
    
    gmsh.write(os.path.join(output_dir, output_prefix + '.dat')) #export mesh to .dat file (only exports 3D mesh)
    if export_vtk:
        gmsh.write(os.path.join(output_dir, output_prefix + '.vtk'))
    gmsh.finalize()
    
    if save_centroids:
        print('Getting centroids ...')
        centroid_elems = np.empty((elems_connectivity.shape[0], 3))
        for i in range(centroid_elems.shape[0]):
            centroid_elems[(elems_connectivity[i,0]-1).astype(int), :] = ((unique_nodes_coords[unique_nodes_coords[:, 0].astype(int) == elems_connectivity[i, 1]][:,1:] +
                                    unique_nodes_coords[unique_nodes_coords[:, 0].astype(int) == elems_connectivity[i, 2]][:,1:] +
                                    unique_nodes_coords[unique_nodes_coords[:, 0].astype(int) == elems_connectivity[i, 3]][:,1:] +
                                    unique_nodes_coords[unique_nodes_coords[:, 0].astype(int) == elems_connectivity[i, 4]][:,1:]) / 4).reshape(3,)
        print('... done')
        print('Exporting the array to a csv file')
        np.savetxt(os.path.join(output_dir,"centroid_elems.csv"), centroid_elems, delimiter=",")
        
    return unique_nodes_coords, elems_connectivity


def write_ansys_inp(nodes,
                    elements,
                    output_dir:str,
                    output_prefix: str):
    """
    
    Exports the given mesh data (nodes and connectivity) as a compatible ANSYS input file.
    It generates two files: nodes.inp and elements.inp, which contain
    the node coordinates and the connectivity, respectively.
    
    Parameters
    ----------
    nodes : numpy array
        Array (Nnodes x 4) containing the node number and coordinates.
        
    elements : numpy array
        Array (Nelements x 5) containing theelement number and node connectivity.
    
    output_prefix : string
        Prefix for output files
        
    output_dir : string
        Destination folder
    
    Returns
    -------
    centroid_elems : numpy array
        Array (Nx3) containig the coordinates of the centroids of the elements.
    """

    prefix = 'Writing nodes:'
    vis.progressbar(0, nodes.shape[0], prefix=prefix, suffix='complete', length=10)
    with open(os.path.join(output_dir,'mesh_nodes.inp'), 'w') as txt_nodes:
        for i in range(nodes.shape[0]):
            vis.progressbar(i + 1, nodes.shape[0], prefix=prefix, suffix='complete', length=10)
            txt_nodes.write('n,' + str(int(nodes[i, 0])) + ',' + str(nodes[i, 1]) + ',' + str(nodes[i, 2]) + ',' + str(
                nodes[i, 3]) + "\n")
    txt_nodes.close()
    prefix = 'Writing elements:'
    vis.progressbar(0, elements.shape[0], prefix=prefix, suffix='complete', length=10)
    with open(os.path.join(output_dir,'mesh_elements.inp'), 'w') as txt_elements:
        for i in range(elements.shape[0]):
            vis.progressbar(i + 1, elements.shape[0], prefix=prefix, suffix='complete', length=10)
            txt_elements.write('en,' + str(int(elements[i, 0])) + ',' + str(int(elements[i, 1])) + ',' +
                               str(int(elements[i, 2])) + ',' + str(int(elements[i, 3])) + ',' + str(
                int(elements[i, 4])) + "\n")
    return 

def write_abaqus_inp(nodes,
                    elements,
                    output_dir:str,
                    output_prefix: str):
    """
    
    Exports the given mesh data (nodes and connectivity) as a compatible ANSYS input file.
    It generates two files: nodes.inp and elements.inp, which contain
    the node coordinates and the connectivity, respectively.
    
    Parameters
    ----------
    nodes : numpy array
        Array (Nnodes x 4) containing the node number and coordinates.
        
    elements : numpy array
        Array (Nelements x 5) containing theelement number and node connectivity.
    
    output_prefix : string
        Prefix for output files
        
    output_dir : string
        Destination folder
    
    Returns
    -------
    centroid_elems : numpy array
        Array (Nx3) containig the coordinates of the centroids of the elements.
    """

    prefix = 'Writing mesh (nodes):'
    elements[:, 0] = (elements[:, 0] - elements[0, 0]) + 1
    with open(os.path.join(output_dir,'mesh_aabqus.inp'), 'w') as mesh_txt:
        mesh_txt.write('*Node\n')
        for i in range(nodes.shape[0]):
            vis.progressbar(i + 1, nodes.shape[0], prefix=prefix, suffix='complete', length=10)
            mesh_txt.write('n,' + str(int(nodes[i, 0])) + ',' + str(nodes[i, 1]) + ',' + str(nodes[i, 2]) + ',' + str(
                nodes[i, 3]) + "\n")
        mesh_txt.write('*Element, type=C3D4\n')
        prefix = 'Writing mesh (nodes):'
        for i in range(elements.shape[0]):
            vis.progressbar(i + 1, elements.shape[0], prefix=prefix, suffix='complete', length=10)
            mesh_txt.write(str(int(elements[i, 0])) + ',' + str(elements[i, 1]) + ',' + str(elements[i, 2]) + ',' + str(
                elements[i, 3])  + ',' + str(elements[i, 4]) + "\n")
    return 