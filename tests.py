import numpy as np
from mesh import Mesh
from specfem3d import SPECFEM3D

ERROR_THRESHOLD = 10**(-5)


def initialise_BA_test_scenario(path='dummy_input/input', fname='block',  nondim=True):
    # from PSEM file
    ngllx = 3
    nglly = 3
    ngllz = 3
    ngnode = 8

    num_materials = 1 # should be loaded from file.

    input_path = path
    fname = fname
    # Read input:
    m = Mesh(input_path, fname, ngllx, nglly, ngllz, ngnode,
             ISPOT_DOF=False, ISNONDIMENSIONAL=nondim)
    m.read_input()
    m.calc_model_coord_extents()
    m._calc_gll1D()
    m.hex2spec()
    m.initialise_dof()
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
    m.compute_max_elementsize()
    m.determine_solver()
    return m



# _____________________________________________________________________________________________________________________
def test_BA_poissons1():
    # Test 1 is an initial test to check that the analytical result for the poisson term
    # is achieved for the SEM formulation. Details are written in the SL latex/overleaf doc.

    print("Testing Poissons Term in BA calculation...")
    # Get initialised mesh setup:
    m = initialise_BA_test_scenario(nondim=True, path='dummy_input/input', fname='block')

    # Run initialisation for SPECFEM3D  object
    print("     Create SPECFEM3D object")
    s = SPECFEM3D(m)
    s.initialise()

    # Test here is for phi to be equal to xy + zx + y^2
    # Test function here is = x + y + 2*z

    # Integral should produce a value of 0.5 * DX * DY * DZ (3 DX + 3 DY + DZ) where DX is x2 - x1 etc etc


    # ------------------------------ SEM solution ------------------------------------------------------------:
    #   We have our coefficient matrix as s.P and this is then multiplied by the Phi
    #   we need to calculate phi values at each gll point:

   # In this case we need to calculate the P matrix with a test function that is not 1:
    tf = np.zeros((s.m.ngll, s.m.nelem))


    # Calculate test function and phi_dot at each GLL point
    for ielem in range(s.m.nelem):
        for igll in range(s.m.ngll):
            num    = m.g_num[igll, ielem] - 1
            coords = m.g_coord[:,num]

            x = coords[0]
            y = coords[1]
            z = coords[2]

            tf[igll, ielem] = x + y + 2*z

            s.Phi_global[num] = x*y  + z*x  + y**2


    # Now we can re-calculate P (elemental) with the test function values:
    # Flag means that tf values are parsed/included
    s._calc_BF_poissons_term(tf=tf)

    # Now can assemble the global matrix:
    s._assemble_poisson_global()


    # Whole SEM integral is then:
    vector = np.matmul(s.P_global, s.Phi_global)
    sumresult = np.sum(vector)

    # -------------------------------------  Analytical solution: ----------------------------------------
    x1 = m.xmin * m.nondim_L
    x2 = m.xmax * m.nondim_L

    y1 = np.min(m.ycoord) * m.nondim_L
    y2 = np.max(m.ycoord) * m.nondim_L

    z1 = np.min(m.zcoord) * m.nondim_L
    z2 = np.max(m.zcoord) * m.nondim_L

    integral = 1.5 * (x2 ** 2 - x1 ** 2) * (y2 - y1) * (z2 - z1) + 1.5 * (x2 - x1) * (y2 ** 2 - y1 ** 2) * (
                z2 - z1) + 0.5 * (x2 - x1) * (y2 - y1) * (z2 ** 2 - z1 ** 2)
    print("     Analytical integral value:", integral)
    # -------------------------------------  Analytical solution: ----------------------------------------



    # ---------- OUTPUT RESULTS ---------
    err_percent = np.abs((sumresult - integral) / integral * 100)
    print('     SEM integral:', sumresult)
    print(f'    *** Relative error: {err_percent} % ***')
    assert(err_percent < ERROR_THRESHOLD)

    # ---------- OUTPUT RESULTS ---------

# _____________________________________________________________________________________________________________________




