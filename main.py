import numpy as np
from mesh import Mesh
# from PSEM file
ngllx = 3
nglly = 3
ngllz = 3
ngnode = 8



num_materials = 1 # should be loaded from file.

input_path = 'dummy_input/input'
fname = 'block'

# Read input:
m = Mesh(input_path, fname, ngllx, nglly, ngllz, ngnode,
         ISPOT_DOF=False)


m.read_input()

m.calc_model_coord_extents()

m._calc_gll1D()

m.hex2spec()

m.initialise_dof()     # Set to false for this example.

m.set_element_dof()

m.prepare_hex()

m.prepare_hexface()

m.prepare_integration()

m.prepare_integration2d()

m.set_model_properties()

m.set_nondimensional_params()

m.calc_nondimensionalisation_vals()

m.prepare_free_surface()

m.apply_nondimensionalisation()

