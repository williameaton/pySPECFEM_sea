import numpy as np
from mesh import Mesh
# from PSEM file
ngllx = 3
nglly = 3
ngllz = 3
ngnode = 8

input_path = 'dummy_input/input'
fname = 'block'

# Read input:
m = Mesh(input_path, fname, ngllx, nglly, ngllz, ngnode)

m.hex2spec()

m.initialise_dof(ISPOT_DOF=False) # Set to false for this example.

m.set_element_dof(ISPOT_DOF=False)

m.prepare_hex()
m.prepare_hexface()

m.prepare_integration()

print()



