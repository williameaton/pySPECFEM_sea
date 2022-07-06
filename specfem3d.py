import numpy as np

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

    def initialise(self):

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


