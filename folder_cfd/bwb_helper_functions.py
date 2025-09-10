import numpy as np
import csdl_alpha as csdl
import lsdo_geo as lg



from pathlib import Path
import pickle
def write_geometry_pickle(geometry, geometry_file_path):

    with open(geometry_file_path, 'wb+') as handle:
        geometry_copy = geometry.copy()
        for i, function in geometry.functions.items():
            function_copy = function.copy()
            function_copy.coefficients = function.coefficients.value.copy()
            geometry_copy.functions[i] = function_copy

        pickle.dump(geometry_copy, handle, protocol=pickle.HIGHEST_PROTOCOL)
 

def read_geometry_pickle(geometry_file_path):
    
    with open(geometry_file_path, 'rb') as handle:
        function_set = pickle.load(handle)
        for function in function_set.functions.values():
            function.coefficients = csdl.Variable(value=function.coefficients)

    return lg.Geometry(functions=function_set.functions, function_names=function_set.function_names, 
                                name=function_set.name, space=function_set.space)


def read_simple_pickle(file_path):
    with open(file_path, 'rb') as handle:
        contents = pickle.load(handle)

    return contents


def write_simple_pickle(var_to_write, file_path):
    with open(file_path, 'wb+') as handle:
        var_to_write_copy = var_to_write.copy()
        pickle.dump(var_to_write_copy, handle, protocol=pickle.HIGHEST_PROTOCOL)



from mpi4py import MPI
def gather_array_to_rank0(x_local: np.ndarray, comm: MPI.Comm = MPI.COMM_WORLD):
    """
    Gather local arrays to rank 0.
    
    Returns on rank 0:
        - x_full: (N_total, dim) array
        - sizes: list of local row counts from each rank
        - index_ranges: list of (start, stop) tuples for each rank's slice
    
    On other ranks:
        - x_full is None
        - sizes and index_ranges are available for consistency
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    local_npts = np.array([x_local.shape[0]], dtype=np.int32)
    sizes = np.zeros(size, dtype=np.int32)
    comm.Allgather([local_npts, MPI.INT], [sizes, MPI.INT])

    dim = x_local.shape[1]
    counts = sizes * dim
    displacements = np.insert(np.cumsum(counts), 0, 0)[:-1]

    # Compute start/stop indices for slicing back full array
    starts = np.insert(np.cumsum(sizes), 0, 0)[:-1]
    stops = starts + sizes
    index_ranges = list(zip(starts, stops))

    sendbuf = x_local.flatten()
    recvbuf = None
    if rank == 0:
        total_count = np.sum(counts)
        recvbuf = np.empty(total_count, dtype=np.float64)

    comm.Gatherv(sendbuf, (recvbuf, counts, displacements, MPI.DOUBLE), root=0)

    if rank == 0:
        x_full = recvbuf.reshape((-1, dim))

        return x_full, sizes, index_ranges
    else:
        return None, sizes, index_ranges
    

# from scipy.spatial import cKDTree    
# def gather_array_to_rank0(x_local: np.ndarray,
#                           comm: MPI.Comm = MPI.COMM_WORLD,
#                           tol: float = 1e-12,
#                           enforce_global_order: bool = True):
#     """
#     MPI gather with optional global ordering.

#     Returns:
#         On rank 0:
#             - x_full: (N_total, dim) array (globally ordered if enforce_global_order=True)
#             - sizes: list of local row counts from each rank
#             - index_ranges: list of (start, stop) tuples for each rank's original slice
#         On other ranks:
#             - x_full is None (unless enforce_global_order is True)
#             - sizes and index_ranges are always returned
#     """
#     rank = comm.Get_rank()
#     size = comm.Get_size()

#     local_npts = np.array([x_local.shape[0]], dtype=np.int32)
#     sizes = np.zeros(size, dtype=np.int32)
#     comm.Allgather([local_npts, MPI.INT], [sizes, MPI.INT])

#     dim = x_local.shape[1]
#     counts = sizes * dim
#     displacements = np.insert(np.cumsum(counts), 0, 0)[:-1]

#     # Original slicing ranges
#     starts = np.insert(np.cumsum(sizes), 0, 0)[:-1]
#     stops = starts + sizes
#     index_ranges = list(zip(starts, stops))

#     sendbuf = x_local.flatten()
#     recvbuf = None
#     if rank == 0:
#         total_count = np.sum(counts)
#         recvbuf = np.empty(total_count, dtype=np.float64)

#     comm.Gatherv(sendbuf, (recvbuf, counts, displacements, MPI.DOUBLE), root=0)

#     x_full = None
#     if rank == 0:
#         x_full = recvbuf.reshape((-1, dim))

#         if enforce_global_order:
#             # ----- Step 1: tolerance-based rounding -----
#             x_rounded = np.round(x_full / tol) * tol

#             # ----- Step 2: unique + deterministic lexicographic sort -----
#             _, unique_idx = np.unique(x_rounded, axis=0, return_index=True)
#             order = np.lexsort(x_full[unique_idx].T)
#             ordered_idx = unique_idx[order]

#             # ----- Step 3: reorder x_full -----
#             x_full = x_full[ordered_idx]

#     # ----- Step 4: Broadcast global ordered array to all ranks if needed -----
#     if enforce_global_order:
#         # Broadcast shape first
#         shape = np.array([0, 0], dtype=np.int32)
#         if rank == 0:
#             shape[:] = x_full.shape
#         comm.Bcast(shape, root=0)

#         # Allocate on other ranks
#         if rank != 0:
#             x_full = np.empty((shape[0], shape[1]), dtype=np.float64)

#         comm.Bcast(x_full, root=0)

#         # ----- Step 5: Reorder local x_local to match global order -----
#         tree = cKDTree(x_full)
#         _, match_idx = tree.query(x_local, k=1)
#         local_order = np.argsort(match_idx)
#         x_local[:] = x_local[local_order]

#     return x_full, sizes, index_ranges




import lsdo_function_spaces as lfs
from lsdo_geo.core.parameterization.free_form_deformation_functions import (
    construct_tight_fit_ffd_block,construct_ffd_block_around_entities,construct_ffd_block_from_corners
)
from lsdo_geo.core.parameterization.volume_sectional_parameterization import (
    VolumeSectionalParameterization,
    VolumeSectionalParameterizationInputs
)
from lsdo_geo.core.parameterization.parameterization_solver import ParameterizationSolver, GeometricVariables
import time

def setup_geometry(geometry: lg.Geometry, geometry_values_dict):

    num_ffd_coefficients_chordwise       = geometry_values_dict['num_ffd_coefficients_chordwise']
    percent_change_in_thickness_dof      = geometry_values_dict['percent_change_in_thickness_dof']
    normalized_percent_camber_change_dof = geometry_values_dict['normalized_percent_camber_change_dof']

    
    # region Create Parameterization Objects
    # num_ffd_coefficients_chordwise = 5
    num_ffd_sections               = 2  # Symmetry boundaries (left, right)
    ffd_block = construct_ffd_block_around_entities(entities=geometry, 
                                                    num_coefficients=(num_ffd_coefficients_chordwise, num_ffd_sections, 2), degree=(3,1,1))

    # ffd_block.plot()
    ffd_sectional_parameterization = VolumeSectionalParameterization(
        name="ffd_sectional_parameterization",
        parameterized_points=ffd_block.coefficients,    # ffd_block.coefficients.shape = (5, 2, 2, 3)
        principal_parametric_dimension=1,
    )
    # ffd_sectional_parameterization.plot()


    # region Evaluate Inner Parameterization Map To Define Forward Model For Parameterization Solver
    sectional_parameters = VolumeSectionalParameterizationInputs()
    ffd_coefficients     = ffd_sectional_parameterization.evaluate(sectional_parameters, plot=False)


    # Apply shape variables (NEW) : (1) THICKNESS
    original_block_thickness = ffd_block.coefficients.value[0, 0, 1, 2] - ffd_block.coefficients.value[0, 0, 0, 2] # normal-thickness  

    percent_change_in_thickness = csdl.Variable(shape=(num_ffd_coefficients_chordwise, num_ffd_sections), value=0.) # (5,2)

    # percent_change_in_thickness_dof = csdl.Variable(shape=(num_ffd_coefficients_chordwise-2,),
    #                                                     value=np.array([0,0,0])) 

    percent_change_in_thickness = percent_change_in_thickness.set(csdl.slice[1:-1,0], percent_change_in_thickness_dof)

    percent_change_in_thickness = percent_change_in_thickness.set(csdl.slice[1:-1,1], percent_change_in_thickness_dof)

    delta_block_thickness = (percent_change_in_thickness / 100) * original_block_thickness
    thickness_upper_translation = 1/2 * delta_block_thickness
    thickness_lower_translation = -thickness_upper_translation

    ffd_coefficients = ffd_coefficients.set(csdl.slice[:,:,1,2], ffd_coefficients[:,:,1,2] + thickness_upper_translation)
    ffd_coefficients = ffd_coefficients.set(csdl.slice[:,:,0,2], ffd_coefficients[:,:,0,2] + thickness_lower_translation)


    # Parameterize camber change as normalized by the original block (kind of like chord) length (NEW) : (2) CAMBER
    normalized_percent_camber_change = csdl.Variable(shape=(num_ffd_coefficients_chordwise, num_ffd_sections),  value=0.)

    # normalized_percent_camber_change_dof = csdl.Variable(shape=(num_ffd_coefficients_chordwise-2,),
    #                                                     value=np.array([0,0,0]))

    normalized_percent_camber_change = normalized_percent_camber_change.set(csdl.slice[1:-1, 0],
                                                                            normalized_percent_camber_change_dof)
    normalized_percent_camber_change = normalized_percent_camber_change.set(csdl.slice[1:-1, 1],         
                                                                            normalized_percent_camber_change_dof)

    block_length     = ffd_block.coefficients.value[1, 0, 0, 0] - ffd_block.coefficients.value[0, 0, 0, 0]

    camber_change    = (normalized_percent_camber_change / 100) * block_length
    ffd_coefficients = ffd_coefficients.set(csdl.slice[:,:,:,2], 
                                            ffd_coefficients[:,:,:,2] + csdl.expand(camber_change, (num_ffd_coefficients_chordwise, num_ffd_sections, 2), 'ij->ijk'))

    geometry_coefficients = ffd_block.evaluate_ffd(coefficients=ffd_coefficients, plot=False)
    geometry.set_coefficients(geometry_coefficients) 

    # geometry.plot(opacity=0.5,color='green',show=PLOT_CONTROL)


    # Roation effect (sort of pitch angle) : not consdiered as design variable
    rotation_axis = np.array([0., 1., 0.])
    rotation_origin = geometry.evaluate(geometry.project(np.array([0.0, 0.0, 0.0])))
    # rotation_angle = 15    
    rotation_angle = 0 #KDH
    geometry.rotate(rotation_origin, rotation_axis, rotation_angle, units='degrees')



    return geometry
