# ===============================
# region PACKAGES
# ===============================
import numpy as np
import sys
import os
import time
import pickle
from pathlib import Path

# MPI
from mpi4py import MPI

# CSDL packages
import csdl_alpha as csdl
import lsdo_function_spaces as lfs
import lsdo_geo

# Optimization
from modopt import CSDLAlphaProblem
from modopt import PySLSQP

# IDWarp and DAFoam
from csdl_idwarp import DAFoamMeshWarper
from csdl_dafoam import instantiateDAFoam, DAFoamFunctions, DAFoamSolver, compute_dafoam_input_variables
import standard_atmosphere_model as sam

# BWB specific
from bwb_helper_functions import setup_geometry, read_geometry_pickle, write_geometry_pickle, gather_array_to_rank0, read_simple_pickle, write_simple_pickle

# Plotting
from vedo import Points, show
import matplotlib.pyplot as plt

# Hashing (for file name generation)
import hashlib

#---- DEBUGGING TOOLS ----
import faulthandler
faulthandler.enable()
os.environ["PETSC_OPTIONS"] = "-malloc_debug"
#-------------------------


# --------------------- lsdo_geo ---------------------
import lsdo_geo
import lsdo_function_spaces as lfs

from lsdo_geo.core.parameterization.free_form_deformation_functions import (
    construct_tight_fit_ffd_block,construct_ffd_block_around_entities
)
from lsdo_geo.core.parameterization.volume_sectional_parameterization import (
    VolumeSectionalParameterization,
    VolumeSectionalParameterizationInputs
)
from lsdo_geo.core.parameterization.parameterization_solver import ParameterizationSolver, GeometricVariables



# ===============================
# region USER INPUT
# ===============================
# Geometry
geometry_directory        =  os.path.join(os.getcwd(), 'geometry_naca0012/')

stp_file_name             = 'airfoil_transonic_unitspan_2.stp'
geometry_pickle_file_name = 'airfoil_transonic_unitspan_2_stored_import.pickle'

# MPI and timing
comm           = MPI.COMM_WORLD
TIMING_ENABLED = True  # True if we want timing printed for the CSDL operations

# DAFoam
dafoam_directory = os.path.join(os.getcwd(), 'openfoam_naca0012/')

# Initial/reference values for DAFoam (best to use base conditions)
U0        = 238.0         # used for normalizing CD and CL
p0        = 101325.0
T0        = 300.0
nuTilda0  = 4.5e-5
CL_target = 0.5
aoa0      = 3
A0        = 0.1           
rho0      = p0 / T0 / 287 # used for normalizing CD and CL

# region Dafoam options
da_options = {
    "designSurfaces": ["wing"],
    "solverName": "DARhoSimpleCFoam",
    "primalMinResTol": 1.0e-8,
    "primalBC": {
        "U0": {"variable": "U", "patches": ["inout"], "value": [U0, 0.0, 0.0]},
        "p0": {"variable": "p", "patches": ["inout"], "value": [p0]},
        "T0": {"variable": "T", "patches": ["inout"], "value": [T0]},
        "nuTilda0": {"variable": "nuTilda", "patches": ["inout"], "value": [nuTilda0]},
        "useWallFunction": True,
    },
    "function": {
        "CD": {
            "type": "force",
            "source": "patchToFace",
            "patches": ["wing"],
            "directionMode": "parallelToFlow",
            "patchVelocityInputName": "patch_velocity",
            "direction": [1, 0, 0],
            "scale": 1.0 / (0.5 * U0 * U0 * A0 * rho0),
        },
        "CL": {
            "type": "force",
            "source": "patchToFace",
            "patches": ["wing"],
            "directionMode": "normalToFlow",
            "patchVelocityInputName": "patch_velocity",
            "scale": 1.0 / (0.5 * U0 * U0 * A0 * rho0),
        },
    },
    "adjEqnOption": {"gmresRelTol": 1.0e-6, "pcFillLevel": 1, "jacMatReOrdering": "rcm", "useNonZeroInitGuess": False},
    # transonic preconditioner to speed up the adjoint convergence
    "transonicPCOption": 1,
    "normalizeStates": {
        "U": U0,
        "p": p0,
        "T": T0,
        "nuTilda": nuTilda0 * 10.0,
        "phi": 1.0,
    },
    "inputInfo": {
        "aero_vol_coords": {
            "type": "volCoord", 
            "components": ["solver", "function"],
        },
        "patch_velocity": {
            "type": "patchVelocity",
            "patches": ["inout"],
            "flowAxis": "x",
            "normalAxis": "z",
            "components": ["solver", "function"],
        },
        "pressure": {
            "type": "patchVar",
            "varName": "p",
            "varType": "scalar",
            "patches": ["inout"],
            "components": ["solver", "function"],
        },
        "temperature": {
            "type": "patchVar",
            "varName": "T",
            "varType": "scalar",
            "patches": ["inout"],
            "components": ["solver", "function"],
        },
    }
}

# region Mesh options
mesh_options = {
    # "gridFile": os.getcwd() + '/idwarp/',
    "gridFile": "./",
    "fileType": "OpenFOAM",
    "symmetryPlanes": [],
}



# ===============================
# region HELPER FUNCTIONS
# ===============================
# TIMER
from contextlib import contextmanager
# Use this to print the timings for certain lines
timings = {}  # Optional: for logging total times

@contextmanager
def Timer(name):
    if TIMING_ENABLED:
        print(f'Rank {rank}: {name}...', flush=True)
        start = time.time()
        yield
        elapsed = time.time() - start
        print(f'Rank {rank}: {name} elapsed time: {elapsed:.3f} s')
        timings[name] = elapsed
    else:
        yield


# HASHER (for generating filenames)
import hashlib
def hash_array_tol(arr: np.ndarray, tol: float = 1e-8, length: int = 16) -> str:
    """
    Generate a tolerance-aware short hash of a NumPy array.

    Parameters:
        arr (np.ndarray): Input array to hash.
        tol (float): Tolerance for rounding (default: 1e-8).
        length (int): Number of hex characters to return from the hash (default: 16).

    Returns:
        str: A truncated SHA-256 hash of the rounded array.
    """
    # Round the array to the given tolerance
    rounded = np.round(arr / tol) * tol
    # Hash the byte representation of the rounded array
    byte_repr = rounded.astype(np.float64).tobytes()
    full_hash = hashlib.sha256(byte_repr).hexdigest()
    return full_hash[:length]

# ===============================
# region SETUP
# ===============================
# MPI information
rank                        = comm.Get_rank()
comm_size                   = comm.Get_size()

# region DAFoam instance
dafoam_instance             = instantiateDAFoam(da_options, comm, dafoam_directory, mesh_options)
x_surf_dafoam_initial_mpi   = dafoam_instance.getSurfaceCoordinates(dafoam_instance.designSurfacesGroup)
# x_vol_dafoam_initial_mpi    = dafoam_instance.xv0
 
x_vol_dafoam                = dafoam_instance.xv0

local_n_surf  = x_surf_dafoam_initial_mpi.shape[0]
local_n_vol   = x_surf_dafoam_initial_mpi.shape[0]

# Gathering surface mesh to rank 0 (need to do this to avoid 'no-element' ranks in the projection
# and geometry evaluation functions)
(x_surf_dafoam_initial, 
x_surf_dafoam_initial_size,
x_surf_dafoam_initial_indices) = gather_array_to_rank0(x_surf_dafoam_initial_mpi, comm)

# Get hash for surface mesh projection file read/write (broadcast to other ranks)
if rank == 0:
    x_surf_hash = hash_array_tol(x_surf_dafoam_initial)
else:
    x_surf_hash = None
x_surf_hash = comm.bcast(x_surf_hash, root=0)

# region File paths
geometry_pickle_file_path         = Path(geometry_directory)/geometry_pickle_file_name
stp_file_path                     = Path(geometry_directory)/stp_file_name
surface_mesh_projection_file_path = Path(dafoam_directory)/f'projected_surface_mesh_{x_surf_hash}.pickle'


# ===============================
# region CSDL RECORDER
# ===============================
# recorder 
recorder = csdl.Recorder(inline=True, debug=True)
recorder.start()

# This section checks to see if a geometry object has already been generated.
#   If so, read the pickle file
#   If not
#       If mpi rank is 0
#           Compute geometry and write pickle file
#       If mpi rank is not 0
#           Wait for rank 1 to finish
#           Read file   

#region Geometry setup I
if geometry_pickle_file_path.is_file():
    with Timer(f'reading geometry'):
        geometry = read_geometry_pickle(geometry_pickle_file_path)
        
else:
    if rank == 0:
        print('No geometry pickle file found.')
        with Timer('importing geometry'):
            geometry = lsdo_geo.import_geometry(stp_file_path,
                                                parallelize=False)
        with Timer('pickling geometry'):
            write_geometry_pickle(geometry, geometry_pickle_file_path)
            
    # Wait for root rank to finish writing
    comm.Barrier()
    if rank != 0:
        with Timer(f'reading geometry'):
            geometry = read_geometry_pickle(geometry_pickle_file_path)   


if rank == 0:
    undeformed_plot = geometry.plot(opacity=0.5,color='yellow',show=False)

comm.Barrier()


# ===== ORIGINAL =====
# region Surface mesh projection
# Now do we do the same check for the surface mesh projection
if surface_mesh_projection_file_path.is_file():
    if rank == 0:
        print('Found surface mesh projection pickle!')
    projected_surf_mesh_dafoam = read_simple_pickle(surface_mesh_projection_file_path)

else:
    if rank == 0:
        print('No projected surface mesh file found.')
        with Timer('projecting on surface mesh'):
            projected_surf_mesh_dafoam = geometry.project(
                x_surf_dafoam_initial, 
                grid_search_density_parameter = 1,      # 1     (ORIGINAL)
                projection_tolerance          = 1e-3,   # 1.e-3 (ORIGINAL)
                grid_search_density_cutoff    = 50,     # 20    (ORIGINAL) 50
                force_reprojection            = False,
                plot                          = True   
            )

        print('Writing surface mesh projection pickle...')
        write_simple_pickle(projected_surf_mesh_dafoam, surface_mesh_projection_file_path)
        print('Done!')

    comm.Barrier()
    if rank != 0:
        projected_surf_mesh_dafoam = read_simple_pickle(surface_mesh_projection_file_path)

# region Design variables
# 4 cores
# ============================ Design variables (2D AIRFOIL) ===========================
num_ffd_coefficients_chordwise=5

percent_change_in_thickness_dof = csdl.Variable(shape=(num_ffd_coefficients_chordwise-2,),
                                                        value=np.array([0,0,0]), name="thickness") 

normalized_percent_camber_change_dof = csdl.Variable(shape=(num_ffd_coefficients_chordwise-2,),
                                                    value=np.array([0,0,0]), name="camber")

geometry_values_dict = {'num_ffd_coefficients_chordwise'      : num_ffd_coefficients_chordwise,
                        'percent_change_in_thickness_dof'     : percent_change_in_thickness_dof,
                        'normalized_percent_camber_change_dof': normalized_percent_camber_change_dof}

# Set as geometric design variables: thickness, camber
percent_change_in_thickness_dof.set_as_design_variable(lower=-100, upper=100, scaler=0.01)
normalized_percent_camber_change_dof.set_as_design_variable(lower=-50, upper=50, scaler=0.01)

# ===== ORIGINAL =====
# region Geometry setup II
with Timer(f'setting up geometry'):
    # Had to "serialize" this because I was getting race conditions in cache I/O
    for r in range(comm_size):
        comm.Barrier()
        if rank == r:
            geometry = setup_geometry(geometry, geometry_values_dict)
            
            if rank == 0:
                deformed_plot = geometry.plot(opacity=0.5,color='red', additional_plotting_elements=undeformed_plot, show=True) 
        comm.Barrier()

with Timer(f'evaluating geometry component'):
    x_surf_dafoam_full = geometry.evaluate(projected_surf_mesh_dafoam, plot=False)

comm.Barrier()

# test_norm = csdl.norm(x_surf_dafoam_full)
# test_norm.set_as_objective()


# ===================================================================================== COMMENT OUT ===========
# Flight condition variables
flight_conditions_group                 = csdl.VariableGroup()

# flight_conditions_group.mach_number     = csdl.Variable(value=0.6, name="mach_number") # Set rither Mach number or freestream
flight_conditions_group.airspeed_m_s    = csdl.Variable(value=238, name="air speed")

flight_conditions_group.angle_of_attack = csdl.Variable(value=aoa0, name="angle_of_attack")
flight_conditions_group.altitude_m      = csdl.Variable(value=0., name="altitude (m)")

# Atmospheric condition variables
ambient_conditions_group = sam.compute_ambient_conditions_group(flight_conditions_group.altitude_m)

# Design variables
flight_conditions_group.angle_of_attack.set_as_design_variable(lower=0, upper=10) 
# flight_conditions_group.airspeed_m_s.set_as_design_variable(lower=220, upper=260)

with csdl.experimental.mpi.enter_mpi_region(rank, comm) as mpi_region:

    # region Surface mesh distribution
    i0, i1          = x_surf_dafoam_initial_indices[rank]
    x_surf_dafoam   = x_surf_dafoam_full[i0:i1,:]

    x_surf_dafoam   = x_surf_dafoam.flatten()
    # print(f"---x_surf_dafoam.shape={x_surf_dafoam.shape}")

    # region IDWarp and DAFoam
    idwarp_model    = DAFoamMeshWarper(dafoam_instance)
    x_vol_dafoam    = idwarp_model.evaluate(x_surf_dafoam)

    flight_conditions_group.angle_of_attack = mpi_region.split_custom(flight_conditions_group.angle_of_attack, split_func = lambda x:x)

    # DAFoam input variable generation
    # Generate our DAFoam CSDL input variable group 
    # (this will add airspeed_m_s to the flight conditions group if not already present)
    dafoam_input_variables_group = compute_dafoam_input_variables(dafoam_instance, 
                                                                ambient_conditions_group, 
                                                                flight_conditions_group,
                                                                x_vol_dafoam)

    # DAFoamSolver Implicit component setup and evaluation
    dafoam_solver           = DAFoamSolver(dafoam_instance)
    dafoam_solver_states    = dafoam_solver.evaluate(dafoam_input_variables_group)

    # DAFoamFunctions Explicit component setup and evaluation
    dafoam_functions = DAFoamFunctions(dafoam_instance)
    dafoam_function_outputs = dafoam_functions.evaluate(dafoam_solver_states, 
                                                        dafoam_input_variables_group)

    # region Design variable, constraint, and objective declaration
    # Declaring and naming some variables
    CL = dafoam_function_outputs.CL
    CD = dafoam_function_outputs.CD

    mpi_region.set_as_global_output(CL)
    mpi_region.set_as_global_output(CD)
    

# Objectives
CD.set_as_objective()
CD.add_name('CD')

# Constraints
CL.set_as_constraint(lower=CL_target, upper=CL_target) # equal=CL_target unavailable
CL.add_name('CL')

recorder.stop()

# ===============================
# region SIM
# ===============================
# sim = csdl.experimental.PySimulator(recorder)
sim = csdl.experimental.JaxSimulator(recorder, 
                                    gpu=False, 
                                    save_on_update=False, 
                                    filename='ASO_2DAF_sim', 
                                    output_saved=False)

# Can set design variables here and run sim to test
# sim[root_twist]  = 3*3.14159/180
# sim.run()
# sim.check_totals()

prob      = CSDLAlphaProblem(problem_name=f'ASO_2DAF_rank{rank}', simulator=sim)
optimizer = PySLSQP(prob, solver_options={'maxiter':20, 'acc':1e-5, 'visualize':False})

# optimizer.check_first_derivatives(prob.x0)
# optimizer.check_first_derivatives()

optimizer.solve()
optimizer.print_results()