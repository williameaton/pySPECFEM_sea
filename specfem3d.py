import numba
import numpy as np
from copy import copy
from numba import int32, int64, boolean, float64
from numba import njit, jit, typeof
from timeit import default_timer as timer
from bilinear_form_numba import _calc_BF_fluid_internal_numba, _calc_BF_strain_deviator_numba

def d(i, j):
    # Kronecker delta function
    if i == j:
        return 1
    else:
        return 0




class SPECFEM3D():
    def __init__(self, m):
        # Takes in mesh object
        self.m = m
        self.NST = 6 # stress tensor dim?
        self.nmaxwell = 1


        # IS IT A DISPLACEMENT BC?
        self.isubc     = True
        self.infbc     = False
        self.isfsubc   = False # Is displacement BC defined on the all unique SEM nodes on the free surface.
        self.isstress0 = False # No pre-stress
        self.solver_diagscale = False

    def initialise(self):

        print()
        print("*******************************************************************************************************")
        print("                        INITIALISING SPECFEM3D PYTHON TEMPORARY VERSION                                ")
        print("*******************************************************************************************************")


        # Calc. relaxation time:
        self._calc_relaxation_time()

        # Count number of elastic vs viscoelastic elements:
        self._count_elmts()

        # Now split up the IDs of elastic and viscoelastic into seperate arrays:
        self._split_elas_visco_eids()

        # Now we apply the displacement boundary conditions
        # First need to allocate some arrays
        self._allocate_disp_BC_arrays()

        # Next we 'activate' the degrees of freedom
        self._activate_dof()

        # Not functional right now
        self._assemble_ghosts_gdof()
        self.gdof[self.gdof>0] = 1

        self._apply_bc()

        # Determine number of equations to solve
        self._finalise_dof()

        self._modify_ghost_gdof

        # Now store elemental global dof from nodal dof array:
        self._store_elem_gdof()

        self._calc_prestress()

        self._allocate_load_arrays()

        self._compute_node_valency()

        self._assemble_ghosts_nodal_iscalar()

        self._allocate_free_variables_and_others()

        self._calc_jacobian()


        # Needs a background gravity term for bilinear form (\nabla \Phi):
        self._set_background_gravity()


        # Calculating bilinear form terms:
        print("Calculating bilinear form:")
        self._calc_BF_poissons_term()         # Term 1
        self.calc_BF_bulk_term()              # Term 2
        self._calc_BF_strain_deviator_numba() # Term 3     - Numba version
        self._calc_BF_background_grav_1()     # Term 4.1
        self._calc_BF_background_grav_2()     # Term 4.2
        self._calc_BF_background_grav_3()     # Term 5
        self._calc_BF_fluid_int()             # Term 7     - Numba version


        # Serial Bilinear Form calculations
        #self._calc_BF_strain_deviator()    # Term 3 - serial version
        #self._calc_BF_fluid_int_serial()   # Term 7 - serial version









    def _calc_relaxation_time(self):
        # currently not implemented:
        pass



    def _count_elmts(self):
        # initialise
        self.nelmt_elas      = 0
        self.nelmt_viscoelas = 0

        for i_elmt in range(self.m.nelem):
            mdomain = self.m.mat_domain[self.m.matID[i_elmt]-1]

            if mdomain == 1: # elastic
                self.nelmt_elas += 1
            elif mdomain == 11: # viscoelastic
                self.nelmt_viscoelas += 1
            else:
                raise ValueError("Only elastic/viscoelastic supported rn.")

        assert(self.nelmt_elas + self.nelmt_viscoelas == self.m.nelem)

        # no parallel implementation yet...
        self.tot_nelmt_elas = self.nelmt_elas
        self.max_nelmt_elas = self.nelmt_elas
        self.min_nelmt_elas = self.nelmt_elas

        # no parallel implementation yet...
        self.tot_nelmt_viscoelas = self.nelmt_viscoelas
        self.max_nelmt_viscoelas = self.nelmt_viscoelas
        self.min_nelmt_viscoelas = self.nelmt_viscoelas



    def _split_elas_visco_eids(self):
        # allocate the elastic and viscoelastic ID arrays:
        eid_elas = np.zeros(self.nelmt_elas, dtype=int)
        eid_viscoelas = np.zeros(self.nelmt_viscoelas, dtype=int)

        ielmt_elas = 0
        ielmt_viscoelas = 0

        for i_elmt in range(self.m.nelem):
            mdomain = self.m.mat_domain[self.m.matID[i_elmt] - 1]

            if mdomain == 1: # elastic - will include infinite elems later.
                eid_elas[ielmt_elas] = i_elmt
                ielmt_elas += 1
            elif mdomain == 11:
                eid_viscoelas[ielmt_viscoelas] = i_elmt
                ielmt_viscoelas += 1

        # Now we can initialise the relevant arrays:
        self.elas_e0 = np.zeros((self.NST , self.m.ngll , self.nelmt_viscoelas))
        self.visco_e0 = np.zeros((self.NST , self.nmaxwell, self.m.ngll, self.nelmt_viscoelas))
        self.q0 = np.zeros((self.NST , self.nmaxwell))


    def _allocate_disp_BC_arrays(self):

        self.bcnodalv = np.zeros((self.m.nndof, self.m.nnode))
        self.gdof     = np.zeros((self.m.nndof, self.m.nnode), dtype=int)

        # Infinite element arrays:
        self.infinite_iface     = np.zeros((6, self.m.nelem), dtype=bool)
        self.infinite_face_idir = np.zeros((6, self.m.nelem))

    def _activate_dof(self):

        if self.m.ISDISP_DOF:
            for i_elmt in range(self.m.nelem):
                inodes = self.m.g_num[:,i_elmt]
                imat = self.m.matID[i_elmt]-1
                mdomain = self.m.mat_domain[imat]

                # if domain is elastic:
                if mdomain == 1:    # elastic
                    for l in list(inodes-1):
                        self.gdof[list(self.m.idofu-1), l] = 1
                elif mdomain == 11: # viscoelastic
                    for l in list(inodes - 1):
                        self.gdof[list(self.m.idofu - 1), l] = 1
                else:
                    raise ValueError('Currently not supported mdomain val')

        if self.m.ISPOT_DOF:
            self.gdof[list(self.m.idofphi-1), :]=1


    def _assemble_ghosts_gdof(self):
        pass

    def _apply_bc(self):

        if np.logical_and(self.m.ISDISP_DOF, self.isubc):
            # Load the BC file:

            # For X direction
            self._set_bc_from_file(fp=f"{self.m.path}/{self.m.fname}_ssbcux", dir=0)

            # For Y direction
            self._set_bc_from_file(fp = f"{self.m.path}/{self.m.fname}_ssbcuy", dir=1)

            # For Z direction
            self._set_bc_from_file(fp=f"{self.m.path}/{self.m.fname}_ssbcuz", dir=2)

            if self.isfsubc:
                raise ValueError("currently not implemented.")


        if self.infbc:
            raise ValueError("infinite elements not implemented.")


    def _set_bc_from_file(self, fp, dir):
        # Dir controls the first index of the gdof and bcnodalv arrays

        line0 = self.m._read_desired_line(file=fp, line=0)[0][0]
        space = line0.find(' ')
        bctype = int(line0[:space])
        val    = float(line0[space:])
        if val != 0:
            self.nozero = True
        else:
            self.nozero = False

        if bctype != 2:
            raise ValueError(" Only surface BCs currently implemented")

        line1   = self.m._read_desired_line(file=fp, line=1)[0][0]
        nelpart = int(line1) # number of elements in the boundary


        # Read the actual elements and their faces:
        bc_elems = np.loadtxt(fp, skiprows=2, dtype=int)
        for i in range(nelpart):
            ielmt = bc_elems[i,0]
            iface = bc_elems[i,1] -1

            self.gdof[dir, self.m.g_num[self.m.hexface[iface].node-1, ielmt-1]-1 ] = 0
            self.bcnodalv[dir, self.m.g_num[self.m.hexface[iface].node-1, ielmt-1]-1 ] = val


    def _finalise_dof(self):

        self.neq = 0 # number of equations to solve for

        sh = np.shape(self.gdof)
        for j in range(sh[1]):
            for i in range(sh[0]):
                if self.gdof[i,j] != 0:
                    self.neq+= 1
                    self.gdof[i,j] = self.neq

        print("Total number of equations to solve: ", self.neq)


    def _modify_ghost_gdof(self):
        pass


    def _store_elem_gdof(self):
        # e.g. for each element it converts a, for example, 3 x 27 (3 degrees of freedom if not using gravity and 27 gll
        # points if ngllx=nglly=ngllz=3) into an 81 (self.m.nedof) element array --> making an 81 x num_of_element array
        self.gdof_elmt = np.zeros((self.m.nedof, self.m.nelem), dtype=int)

        for i_elmt in range(self.m.nelem):
            a = self.gdof[:, self.m.g_num[:,i_elmt]-1 ]

            self.gdof_elmt[:, i_elmt] = a.flatten(order='F') # flatten in consistent style to Fortran (column major)


    def _calc_prestress(self):
        if self.isstress0:
            raise ValueError('prestress currently not implemented.')


    def _allocate_load_arrays(self):
        self.slipload = np.zeros(self.neq)
        self.extload  = np.zeros(self.neq)
        self.rhoload  = np.zeros(self.neq)
        self.ubcload  = np.zeros(self.neq)


    def _compute_node_valency(self):
        self.node_valency = np.zeros(self.m.nnode, dtype=int)

        for i in range(self.m.nelem):
            num = self.m.g_num[:, i]
            self.node_valency[num-1] =  self.node_valency[num-1] + 1


    def _assemble_ghosts_nodal_iscalar(self):
        pass

    def _allocate_free_variables_and_others(self):
        # Allocates an assortment of arrays. I believe that they msut be allocated
        # After the assemble_ghosts? hence why I didnt allocate iwth previous function

        # Loads:
        self.load        = np.zeros(self.neq)
        self.bodyload    = np.zeros(self.neq)
        self.viscoload   = np.zeros(self.neq)
        self.resload     = np.zeros(self.neq)

        # Free variables:
        self.u           = np.zeros(self.neq)
        self.du          = np.zeros(self.neq)

        # Stiffness matrix:
        self.kmat          = np.zeros((self.m.nedof, self.m.nedof))
        self.storekmat          = np.zeros((self.m.nedof, self.m.nedof, self.m.nelem))

        # Storage for mass matrix
        self.storemmat          = np.zeros((self.m.nedof, self.m.nelem))
        self.storemmat_global   = np.zeros(self.m.nnode)
        self.storemmat_global2   = np.zeros(self.m.nnode)


    def _allocate_inbuild_preconditioner(self):

        self.storederiv = np.zeros((self.m.ndim, self.m.ngll, self.m.ngll, self.m.nelmt))
        self.storejw    = np.zeros((self.m.ngll, self.m.nelmt))

        self.dprecon    = np.zeros(self.neq)

        # Scaled diagonal preconditioner?
        if self.solver_diagscale:
            self.ndscale = np.zeros(self.neq)






    def _compute_mass_elastic_global_WE(self):
        print(f"Computing mass matrix with {self.m.nndof} degrees of freedom.")

        for elem_ind in range(self.m.nelem):
            gll_ind = 0

            num = self.m.g_num[:, elem_ind]

            for alpha in range(self.m.ngllx):
                for beta in range(self.m.nglly):
                    for gamma in range(self.m.ngllz):

                        glob_node = num[gll_ind]

                        w = self.m.gll_weights[gll_ind]                 # Product of weights
                        rho = self.m.massden_elmt[gll_ind, elem_ind]    # Density for element
                        mass =  w * self.detjac[gll_ind, elem_ind] * rho

                        self.storemmat_global[glob_node-1]  = self.storemmat_global[glob_node-1] + mass

                        gll_ind += 1



    def _set_background_gravity(self):
        # Create g_zero and set all to 1 for now:
        self.g_zero = np.zeros((self.m.ndim, self.m.ngll, self.m.nelem)) + 1




    def _calc_jacobian(self):

        print(f"Computing Jacobian.")

        # Initialise arrays:
        self.jacobian = np.zeros((self.m.ndim, self.m.ndim, self.m.ngll, self.m.nelem)) # 3 x 3 for each gll of each element
        self.detjac   = np.zeros((self.m.ngll, self.m.nelem))
        self.jacinv   = np.zeros((self.m.ndim, self.m.ndim, self.m.ngll, self.m.nelem))

        # Loop through elements:
        for i_elem in range(self.m.nelem):

            # Get element coordinates for the corner nodes, in XYZ space.
            num = self.m.g_num[:, i_elem]
            a = self.m.g_coord[:, num[self.m.hex8_gnode - 1] - 1]
            coord = np.transpose(a)

            for igll in range(self.m.ngll):

                d = self.m.dshape_hex8[:,:,igll]
                jacobian = np.matmul(d, coord)

                self.jacobian[:,:, igll, i_elem]  =  jacobian[:,:]
                self.detjac[igll, i_elem]         =  np.linalg.det(jacobian)
                self.jacinv[:,:, igll, i_elem]    =  np.linalg.inv(jacobian)[:,:]








    def _calc_poisson_term_HNG(self):
        # THIS MATRIX MULTIPLES THE PHI VALUES
        # Using the matrix format that HNG proposed
        # Output matrix will be called H:
        if self.m.nedofphi != 1:
            Warning(f"Calculating solid phi coeff. matrix even tho self.m.nedofphi = {self.m.nedofphi}")

        self.P_HNG = np.zeros(( self.m.nelem, self.m.ngll, self.m.ngll))

        for i_elem in range(self.m.nelem):
            # Matrix wise calc
            for igll in range(self.m.ngll):
                weight = self.m.gll_weights[igll]
                jacw = self.detjac[igll, i_elem] * weight

                deriv = np.matmul(np.transpose(self.m.dlagrange_gll[:, igll, :]), self.jacinv[:, :, igll, i_elem])
                D = np.matmul(deriv, np.transpose(deriv))

                self.P_HNG[i_elem, :,:] += jacw * D



    def _calc_BF_poissons_term(self):
        print("          poissons eqn term...")

        # Equivalent to the matrix formulation down to 13 decimal palces
        self.P = np.zeros((self.m.nelem, self.m.ngll, self.m.ngll))

        for i_elem in range(self.m.nelem):
            for abg in range(self.m.ngll):
                for stv in range(self.m.ngll):
                    quad_sum = 0
                    for abgbars in range(self.m.ngll):
                        w = self.m.gll_weights[abgbars]
                        jacdet = self.detjac[abgbars, i_elem]

                        i_sum = 0
                        for i in range(3):
                            t1 = 0
                            for j in range(3):
                                t1 += self.jacinv[j, i, stv, i_elem] * self.m.dlagrange_gll[j, abgbars, abg]

                            t2 = 0
                            for q in range(3):
                                t2 += self.jacinv[q, i, stv, i_elem] * self.m.dlagrange_gll[q, abgbars, stv]

                            i_sum += t1*t2
                        quad_sum += i_sum * w * jacdet

                    self.P[i_elem, stv, abg] = quad_sum



    def calc_BF_bulk_term(self):
        print("          bulk modulus term...")
        self.K = np.zeros((self.m.nelem, self.m.ngll*3, self.m.ngll*3))

        for i_elem in range(self.m.nelem):

            m = 0 # Index for output matrix ROWS
            for abg in range(self.m.ngll):
                for i in range(3):

                    n = 0 # Index for output matrix COLUMNS
                    for stn in range(self.m.ngll):
                        for j in range(3):

                            bars_sum = 0
                            for abg_bars in range(self.m.ngll):
                                pi          = self.detjac[abg_bars, i_elem] * self.m.gll_weights[abg_bars]
                                kappa       = self.m.bulkmod_elmt[abg_bars, i_elem]
                                jacinv_ii   = self.jacinv[i, i, abg_bars, i_elem]
                                jacinv_jj   = self.jacinv[j, j, abg_bars, i_elem]
                                dlag_i      = self.m.dlagrange_gll[i, abg_bars, abg]
                                dlag_j      = self.m.dlagrange_gll[j, abg_bars, stn]

                                bars_sum += pi * kappa * jacinv_ii * jacinv_jj * dlag_i * dlag_j

                            self.K[i_elem, m,n] = bars_sum

                            n += 1
                    m += 1





    def _calc_BF_strain_deviator_numba(self):
        print("          strain deviator term using numba...")
        self.D = _calc_BF_strain_deviator_numba(detjac        = self.detjac[:,:],
                                                gll_weights   = self.m.gll_weights[:],
                                                ngll          = self.m.ngll,
                                                nelem         = self.m.nelem,
                                                shearmod      = self.m.shearmod_elmt[:,:],
                                                jacinv        = self.jacinv[:,:,:,:],
                                                dlagrange_gll = self.m.dlagrange_gll[:,:,:]
                                               )





    def _calc_BF_background_grav_1(self):
        print("          background gravity term 1...")

        # Background gravity part 1
        self.B = np.zeros((self.m.nelem, self.m.ngll * 3, self.m.ngll * 3))

        for i_elem in range(self.m.nelem):
            m = 0  # Index for output matrix ROWS
            for abg in range(self.m.ngll):
                for i in range(3):
                    n = 0  # Index for output matrix COLUMNS
                    for abg_bars in range(self.m.ngll):
                        for j in range(3):

                            g0  = self.g_zero[i, abg, i_elem]
                            pi  = self.detjac[abg_bars, i_elem] * self.m.gll_weights[abg_bars]
                            rho = self.m.massden_elmt[abg_bars, i_elem]

                            k_sum = 0
                            for k in range(3):
                                k_sum += self.jacinv[k, j, abg_bars, i_elem]*self.m.dlagrange_gll[k, abg_bars, abg]


                            self.B[i_elem, m,n] = k_sum * g0 * pi * rho

                            n += 1
                    m += 1




    def _calc_BF_background_grav_2(self):
        print("          background gravity term 2...")

        # Background gravity part 1
        self.B2 = np.zeros((self.m.nelem, self.m.ngll * 3, self.m.ngll * 3))

        for i_elem in range(self.m.nelem):
            m = 0  # Index for output matrix ROWS
            for abg_bars in range(self.m.ngll):
                for i in range(3):
                    n = 0  # Index for output matrix COLUMNS
                    for stn in range(self.m.ngll):
                        for j in range(3):

                            g0  = self.g_zero[j, stn, i_elem]
                            pi  = self.detjac[abg_bars, i_elem] * self.m.gll_weights[abg_bars]
                            rho = self.m.massden_elmt[abg_bars, i_elem]

                            k_sum = 0
                            for k in range(3):
                                k_sum += self.jacinv[k, i, abg_bars, i_elem]*self.m.dlagrange_gll[k, abg_bars, stn]


                            self.B2[i_elem, m,n] = k_sum * g0 * pi * rho

                            n += 1
                    m += 1
        print("NOTE: Neither background gravity term contains the half outisde the integral in BF")



    def _calc_BF_background_grav_3(self):
        print("          background gravity term 3...")

        # Background gravity part 1
        self.B3 = np.zeros((self.m.nelem, self.m.ngll * 3, self.m.ngll * 3))

        for i_elem in range(self.m.nelem):
            m = 0  # Index for output matrix ROWS
            for abg in range(self.m.ngll):
                for i in range(3):
                    n = 0  # Index for output matrix COLUMNS
                    for stn in range(self.m.ngll):
                        for j in range(3):

                            bars_sum = 0
                            for abg_bars in range(self.m.ngll):
                                pi = self.detjac[abg_bars, i_elem] * self.m.gll_weights[abg_bars]
                                rho = self.m.massden_elmt[abg_bars, i_elem]

                                t1 = self.jacinv[i,i,abg_bars,i_elem] * self.m.dlagrange_gll[i, abg_bars, abg] * \
                                     self.g_zero[j, abg_bars, i_elem]

                                t2 = self.jacinv[j,j,abg_bars,i_elem] * self.m.dlagrange_gll[j, abg_bars, stn] * \
                                     self.g_zero[i, abg_bars, i_elem]

                                bars_sum += (t1+t2)*pi*rho


                            self.B3[i_elem, m, n] = bars_sum

                            n += 1
                    m += 1
        print("NOTE: Doesnt contain the 1/2 outside the integral")



    def _calc_BF_fluid_int(self):
        self.Fnumba = np.zeros((self.m.nelem, self.m.ngll, self.m.ngll))

        print('Term 7: constant terms not defined properly.')
        g_1 = 1        # imagine g^-1 is always 1
        del_n_rho = 1  # imagine del_n rho is always 1

        self.Fnumba[:,:,:] = _calc_BF_fluid_internal_numba(detjac=self.detjac[:, :],
                             gll_weights=self.m.gll_weights[:],
                             ngll = self.m.ngll,
                             nelem=self.m.nelem,
                             g_1 = g_1,
                             del_n_rho = del_n_rho)



        ############################# SERIAL BILINEAR FORM TERMS: #############################

        def _calc_BF_fluid_int_serial(self):
            print("          fluid region gravity term...")
            self.F = np.zeros((self.m.nelem, self.m.ngll, self.m.ngll))

            g_1 = 1  # imagine g^-1 is always 1
            del_n_rho = 1  # imagine del_n rho is always 1

            for i_elem in range(self.m.nelem):
                for abg in range(self.m.ngll):
                    for stn in range(self.m.ngll):
                        for abg_bars in range(self.m.ngll):
                            pi = self.detjac[abg_bars, i_elem] * self.m.gll_weights[abg_bars]
                            self.F[i_elem, abg, stn] += g_1 * del_n_rho * d(abg_bars, abg) * d(abg_bars, stn) * pi




    def _calc_BF_strain_deviator(self):
        self.Dserial = np.zeros((self.m.nelem, self.m.ngll*3, self.m.ngll*3))
        print("          strain deviator term...")

        for i_elem in range(self.m.nelem):
            m = 0  # Index for output matrix ROWS
            for abg in range(self.m.ngll):
                for i in range(3):
                    n = 0  # Index for output matrix COLUMNS
                    for stn in range(self.m.ngll):
                        for j in range(3):
                            bars_sum = 0
                            for abg_bars in range(self.m.ngll):
                                pi = self.detjac[abg_bars, i_elem] * self.m.gll_weights[abg_bars]
                                mu = self.m.shearmod_elmt[abg_bars, i_elem]
                                sum = self._get_D_internal_sum(i=i, j=j, stn=stn, abg=abg, bars=abg_bars, i_elem=i_elem)
                                bars_sum += (sum * pi * mu)

                            self.Dserial[i_elem, m, n] = bars_sum

                            n += 1
                    m += 1
            print(f"Completed {i_elem+1}/{self.m.nelem+1} elements")
        # Apply the half factor:
        self.Dserial = self.Dserial*0.5000000000000



    def _get_D_internal_sum(self, i, j, stn, abg, bars, i_elem):
        internalsum = 0
        for k in range(3):
            for r in range(3):
                for q in range(3):
                    for p in range(3):
                        t1 = ((d(r,i) * self.jacinv[k, q, bars, i_elem]) + (d(q,i) * self.jacinv[k, r, bars, i_elem])
                              - ( (2/3) * d(r,q) * d(i,k) *
                                  self.jacinv[k, i, bars, i_elem])) * self.m.dlagrange_gll[k, bars, abg]
                        t2 = ((d(r,j) * self.jacinv[p, q, bars, i_elem]) + (d(q,j) * self.jacinv[p, r, bars, i_elem])
                              - ( (2/3) * d(r, q) * d(j, p) *
                                  self.jacinv[p, j, bars, i_elem])) * self.m.dlagrange_gll[p, bars, stn]

                        internalsum += t1*t2
        return internalsum





