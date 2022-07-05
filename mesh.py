# Class for the mesh
import csv
from copy import copy
import numpy as np
from GLL import *
from hex_classes import hex_face, hex_face_edge



class Mesh():
    def __init__(self, path, fname, ngllx, nglly, ngllz, ngnode):
        self.path  = path
        self.fname = fname


        self.ndim = 3
        self.nndofu = 3
        self.nndofphi = 1
        self.ngnode = ngnode
        self.nelem = self._get_nelem()
        self.ngnod = 8 # 8 corners in a hexahedra
        self.ngllx = int(ngllx)
        self.nglly = int(nglly)
        self.ngllz = int(ngllz)
        self.ngllxy = self.ngllx*self.nglly
        self.ngllzx = self.ngllx * self.ngllz
        self.ngllyz = self.nglly * self.ngllz
        self.ngll = int(self.ngllx*self.nglly*self.ngllz)
        self.npoints = self.ngll*self.nelem

        self.nenode = copy(self.ngll) # This is the number of element nodes
                                      # where as self.nnode is the number of unique
                                      # nodes

        self._calc_gll1D()

        # Load mesh stuff:
        self.g_num = np.loadtxt(f'{path}/{fname}_connectivity', skiprows=1, dtype=int)
        self._load_coords()
        self._load_free_surface()
        self._load_mat_IDs()
        self._load_mat_list()
        self._load_BCs()


        # Initialise arrays:
        self.massden = np.zeros((self.ngll, self.nelem))
        self.shearmod = np.zeros((self.ngll, self.nelem))
        self.bulkmod = np.zeros((self.ngll, self.nelem))



    def _get_nelem(self):
        # Determine number of elements:
        desired = [0]
        with open(f'{self.path}/{self.fname}_connectivity', 'r') as fin:
            reader = csv.reader(fin)
            nelem = [[int(s) for s in row] for i, row in enumerate(reader) if i in desired]
        fin.close()
        return nelem[0][0]


    def _load_coords(self):
        xcoord = np.loadtxt(f'{self.path}/{self.fname}_coord_x')
        ycoord = np.loadtxt(f'{self.path}/{self.fname}_coord_y')
        zcoord = np.loadtxt(f'{self.path}/{self.fname}_coord_z')

        self.xcoord = xcoord[1:]
        self.ycoord = ycoord[1:]
        self.zcoord = zcoord[1:]

        self.xmin = np.min(self.xcoord)
        self.xmax = np.max(self.xcoord)

        self.g_coord = np.array([self.xcoord, self.ycoord, self.zcoord])



    def _load_free_surface(self):
        desired = [0]
        with open(f'{self.path}/{self.fname}_free_surface', 'r') as fin:
            reader = csv.reader(fin)
            self.NFreeSurface = [[int(s) for s in row] for i, row in enumerate(reader) if i in desired][0][0]
        fin.close()

        assert(self.NFreeSurface <= self.nelem)

        self.free_surface = np.loadtxt(f'{self.path}/{self.fname}_free_surface', skiprows=1, dtype=int)

    def _load_mat_IDs(self):
        matid = np.loadtxt(f'{self.path}/{self.fname}_material_id', dtype=int)
        assert(matid[0]==self.nelem)

        self.matID = matid[1:]


    def _load_mat_list(self):
        materials = []
        # open file in read mode
        with open(f'{self.path}/{self.fname}_material_list', 'r') as read_obj:
            csv_reader = csv.reader(read_obj)
            ctr = 0
            for row in csv_reader:
                if row[0][0] != '#':
                    if ctr ==0:
                        self.nmaterials = int(row[0])
                    else:
                        materials.append(row)
                    ctr +=1

        read_obj.close()

        self.materials = materials


    def _load_BCs(self):
        # open file in read mode
        with open(f'{self.path}/{self.fname}_ssbcux', 'r') as read_obj:
            csv_reader = csv.reader(read_obj)
            ctr = 0
            while ctr < 2:
                for row in csv_reader:
                    if ctr == 0:
                        self.bcx_type = row[0][0]
                        self.bcx_val  = row[0][1]
                    if ctr == 1:
                        self.bcx_nelem = row[0][0]
                    ctr +=1

        self.bcx_elem = np.loadtxt(f'{self.path}/{self.fname}_ssbcux', skiprows=2, dtype=int)

        # open file in read mode
        with open(f'{self.path}/{self.fname}_ssbcuy', 'r') as read_obj:
            csv_reader = csv.reader(read_obj)
            ctr = 0
            while ctr < 2:
                for row in csv_reader:
                    if ctr == 0:
                        self.bcy_type = row[0][0]
                        self.bcy_val = row[0][1]
                    if ctr == 1:
                        self.bcy_nelem = row[0][0]
                    ctr += 1

        self.bcyelem = np.loadtxt(f'{self.path}/{self.fname}_ssbcuy', skiprows=2, dtype=int)


        # open file in read mode
        with open(f'{self.path}/{self.fname}_ssbcuz', 'r') as read_obj:
            csv_reader = csv.reader(read_obj)
            ctr = 0
            while ctr < 2:
                for row in csv_reader:
                    if ctr == 0:
                        self.bcz_type = row[0][0]
                        self.bcz_val = row[0][1]
                    if ctr == 1:
                        self.bcz_nelem = row[0][0]
                    ctr += 1

        self.bczelem = np.loadtxt(f'{self.path}/{self.fname}_ssbcuz', skiprows=2, dtype=int)

    def _calc_gll1D(self):
        [self.gllpx, self.gllwx] = gll(self.ngllx - 1)
        [self.gllpy, self.gllwy] = gll(self.nglly - 1)
        [self.gllpz, self.gllwz] = gll(self.ngllz - 1)



    def initialise_dof(self, ISDISP_DOF=True, ISPOT_DOF=True):
        # Takes in a mesh object
        # Sets the number of DofF and IDs of the nodal DofF


        nndof = 0  # number of DOF per node
        nedofu = 0  # number of element displacement DOF
        nedofphi = 0  # number of element phi DOF
        nedof = 0  # number of element DOF

        # DOF id:
        idofu = np.zeros(3, dtype=int)
        idof = 0
        idofphi = np.zeros(1, dtype=int)

        # Displacement DOF:
        if ISDISP_DOF:
            nndof += self.nndofu               # Add 3 for disp.
            nedofu = self.nndofu * self.nenode # Elemental DOF disp = number of GLL * 3 (per element)
                                          # E.g. 81 if 3 gll in each direction x 3 disp. directions
            nedof += nedofu

            # Basically just keeps a record that there are 3 degrees of freedom
            # for displacement (idof = 3) and that they are indexed [1,2,3]
            for i_dof in range(self.nndofu):
                idof += 1
                idofu[i_dof] = idof  # +1 to be consistent w fortran

            # initialise element DOF for U
            self.edofu = np.zeros(nedofu, dtype=int)

        idof = idofu[self.nndofu - 1]

        if ISPOT_DOF:
            nndof   += self.nndofphi
            nedofphi = self.nndofphi * self.nenode
            nedof   +=  nedofphi

            for i_dof in range(self.nndofphi):
                idof += 1
                idofphi[i_dof-1] = idof

                # initialise element DOF for PHI
                self.edofphi = np.zeros(nedofphi)

        # Save some variables for future use:
        self.nndof = nndof
        self.idof = idof
        self.nedof = nedof

        self.nedofu = nedofu
        self.idofu = idofu

        self.nedofphi  = nedofphi
        self.idofphi = idofphi



    def set_element_dof(self, ISDISP_DOF=True, ISPOT_DOF=True):
        # Sets IDs for the element level DOF for U and Phi so we can map the element
        # matrices

        self.edofu[:] = -9999

        if ISPOT_DOF:
            self.edofphi[:] = -9999

        iu0   = 0
        iphi0 = 0
        iphi = 0
        nu = 0
        iu = np.zeros(self.nndofu, dtype=int) # SOME KIND OF ARRAY

        for i in range(self.ngll):
            if ISDISP_DOF:
                iu[0] = iu0 + 1
                nu += 1
                self.edofu[nu-1] = iu[0]

                for j in range(1, self.nndofu):

                    nu += 1

                    iu[j] = iu[j-1] + 1
                    self.edofu[nu-1] = iu[j]

                iu0 = iu[self.nndofu-1]
                iphi0 = iu[self.nndofu-1]


            if ISPOT_DOF:
                iphi = iphi0 + 1
                self.edofphi[i] = iphi

                iu0 = iphi
                iphi0 = iphi


    def hex2spec(self):
        # allocate xstore, ystore, zstore:
        self.xstore = np.zeros(self.npoints)
        self.ystore = np.zeros(self.npoints)
        self.zstore = np.zeros(self.npoints)

        # Create 3D shape functions stored in self.shape_hex8
        self.shape_function_hex8()

        i_point = 0
        # now fill the xstore, ystore and zstore with coordinates:
        for i_elmt in range(self.nelem):
            for k in range(self.ngllz):
                for j in range(self.nglly):
                    for i in range(self.ngllx):
                        xgll = 0.0
                        ygll = 0.0
                        zgll = 0.0

                        for i_gnod in range(self.ngnod): #ngnod = 8
                            # need to use -1 because python starts at 0 where as g_num goes from 1 to ...
                            xgll += self.shape_hex8[i_gnod, i,j,k]*self.g_coord[0, self.g_num[i_elmt, i_gnod]-1]
                            ygll += self.shape_hex8[i_gnod, i,j,k]*self.g_coord[1, self.g_num[i_elmt, i_gnod]-1]
                            zgll += self.shape_hex8[i_gnod, i,j,k]*self.g_coord[2, self.g_num[i_elmt, i_gnod]-1]

                        self.xstore[i_point] = xgll
                        self.ystore[i_point] = ygll
                        self.zstore[i_point] = zgll

                        i_point+= 1
        # Now need to call get_global:
        # gets ibool indexing from local (gll points) to global points
        self.get_global_WE()


        self.iglob = self.get_global_indirect_addressing()


        # Now we actually create the g_coord and g_num arrays:
        self.g_coord = np.zeros((3, self.nnode))
        self.g_num = np.zeros((self.ngll, self.nelem), dtype=int)
        ipoint = 0
        for i_elmt in range(self.nelem):
            ienode = 0
            for k in range(self.ngllz):
                for j in range(self.nglly):
                    for i in range(self.ngllx):
                        inode = self.iglob[ipoint]

                        self.g_num[ienode, i_elmt] = inode
                        self.g_coord[0, inode-1] = self.xstore[ipoint]
                        self.g_coord[1, inode-1] = self.ystore[ipoint]
                        self.g_coord[2, inode-1] = self.zstore[ipoint]

                        ienode += 1
                        ipoint += 1



    def get_global_WE(self):
        xold = copy(self.xstore)
        yold = copy(self.ystore)
        zold = copy(self.zstore)

        self.xp = copy(xold)
        self.yp = copy(yold)
        self.zp = copy(zold)
        self.iglob = np.zeros(self.npoints, dtype=int)

        # Create two arrays - the first is an array of the present coordinates:
        D = np.zeros((self.npoints, 3))
        D[:,0] = self.xp
        D[:,1] = self.yp
        D[:,2] = self.zp

        # Now need to sort this but keep the original:
        E = D[D[:, 2].argsort()][::-1]
        F = E[E[:, 1].argsort(kind='mergesort')]
        F = F[F[:, 0].argsort(kind='mergesort')]

        # We now want to remove the repeated coordinate values which decimates the size from npoints --> nnode
        uniques = np.unique([tuple(row) for row in F], axis=0)

        # Now these unique sorted coordinates are numbered in ascending order:
        U = np.zeros((len(uniques), 4))
        U[:,:3] = uniques
        U[:, 3] = np.arange(1, len(uniques)+1)

        # Now we need to search each coordinate in the original set D, in the unique listing and get the corresponding
        # ID that is given in the 4th row of U:

        for i in range(self.npoints):
            self.iglob[i] = int(np.where((U[:, :-1] == D[i, :]).all(axis=1))[0][0] + 1)

        self.nnode = len(uniques)


    def get_global_indirect_addressing(self):
        mask_ibool     = np.zeros(self.npoints) - 1
        ibool          = np.zeros(self.npoints, dtype=int)
        copy_ibool_ori = copy(self.iglob)

        inumber = 0
        for i_point in range(self.npoints):
            if mask_ibool[copy_ibool_ori[i_point]-1] == -1:
                inumber += 1
                ibool[i_point] = inumber
                mask_ibool[copy_ibool_ori[i_point]-1] = inumber
            else:
                ibool[i_point] = mask_ibool[copy_ibool_ori[i_point]-1]


        return ibool

    def shape_function_hex8(self):
        # 3D shape function is a 4D array:
        self.shape_hex8 = np.zeros((self.ngnod, self.ngllx, self.nglly, self.ngllz))

        one = 1.0000000
        one_eighth = 0.12500000

        # Create the shape functions:
        for k in range(self.ngllz):
            for j in range(self.nglly):
                for i in range(self.ngllx):
                    xi_p = one + self.gllpx[i]
                    xi_m = one - self.gllpx[i]

                    eta_p = one + self.gllpy[j]
                    eta_m = one - self.gllpy[j]

                    zeta_p = one + self.gllpz[k]
                    zeta_m = one - self.gllpz[k]

                    self.shape_hex8[0, i, j, k] = one_eighth * xi_m * eta_m * zeta_m
                    self.shape_hex8[1, i, j, k] = one_eighth * xi_p * eta_m * zeta_m
                    self.shape_hex8[2, i, j, k] = one_eighth * xi_p * eta_p * zeta_m
                    self.shape_hex8[3, i, j, k] = one_eighth * xi_m * eta_p * zeta_m
                    self.shape_hex8[4, i, j, k] = one_eighth * xi_m * eta_m * zeta_p
                    self.shape_hex8[5, i, j, k] = one_eighth * xi_p * eta_m * zeta_p
                    self.shape_hex8[6, i, j, k] = one_eighth * xi_p * eta_p * zeta_p
                    self.shape_hex8[7, i, j, k] = one_eighth * xi_m * eta_p * zeta_p

        # Check the shape functions and their derivatives:
        for k in range(self.ngllz):
            for j in range(self.nglly):
                for i in range(self.ngllx):
                    sum_shape = 0
                    for i_gnod in range(0, self.ngnod):
                        sum_shape += self.shape_hex8[i_gnod, i,j,k]

                    if np.abs(sum_shape)-1 > 1e-20:
                        raise ValueError("Shape functions larger than tolerance")



    def _swap_all(self, ia,a,b,c,n, ioff):

        iw = copy(ia)
        w  = copy(a)

        for i in range(n):
            ia[i] = copy(iw[self.ind[i]-1])
            a[i]  = copy(w[self.ind[i]-1])

        w = copy(b)

        for i in range(n):
            b[i] = copy(w[self.ind[i]-1])

        w = copy(c)

        for i in range(n):
            c[i] = copy(w[self.ind[i]-1])

        return ia, a, b, c




    def _edge_loop(self, j_low, j_high, i_low, i_high, iedge, iface, ind_arr, nx, loop_ij):
        for j in range(j_low, j_high):
            jm1 = j - 1 + 1

            for i in range(i_low, i_high):

                if loop_ij == 'j':
                    looper = j
                elif loop_ij == 'i':
                    looper = i
                else:
                    raise ValueError(f"must be i or  but you gave: {looper}")

                ind_arr[looper] = jm1*(nx) + i+1

        self.hexface_edge[iface-1][iedge].set_fnode(copy(ind_arr))
        self.hexface_edge[iface-1][iedge].set_node(copy(self.hexface[iface-1].node[list(ind_arr-1)]))






    def prepare_hex(self):

        self.hex8_gnode = np.zeros(8, dtype=int)

        self.hex8_gnode[0] = 1
        self.hex8_gnode[1] = copy(self.ngllx)
        self.hex8_gnode[2] = copy(self.ngllxy)
        self.hex8_gnode[3] = self.hex8_gnode[2]-self.ngllx+1

        self.hex8_gnode[4] = (self.ngllz-1)*self.ngllxy+1
        self.hex8_gnode[5] = self.hex8_gnode[4]+self.ngllx-1
        self.hex8_gnode[6] = self.ngll
        self.hex8_gnode[7] = self.hex8_gnode[6]-self.ngllx+1

        print("Completed prepare_hex.")


    def prepare_hexface(self):
        # Allocate 6 hexfaces:
        # hexface[0] = ZX        hexface[3] = YZ
        # hexface[1] = YZ        hexface[4] = XY
        # hexface[2] = ZX        hexface[5] = XY

        # Create array called hexface which holds 6 instances of
        # the hex_face class:
        self.hexface = [hex_face(node_dim=self.ngllzx),
                        hex_face(node_dim=self.ngllyz),
                        hex_face(node_dim=self.ngllzx),
                        hex_face(node_dim=self.ngllyz),
                        hex_face(node_dim=self.ngllxy),
                        hex_face(node_dim=self.ngllxy)]

        inode = 0
        i1 = 0; i2 = 0; i3=0; i4=0; i5=0; i6=0

        for k in range(self.ngllz):
            for j in range(self.nglly):
                for i in range(self.ngllx):
                    inode += 1

                    if i == 0:
                        i4 += 1
                        self.hexface[3].node[i4-1] = inode

                    if i == self.ngllx-1:
                        i2 += 1
                        self.hexface[1].node[i2-1] = inode

                    if j == 0:
                        i1 += 1
                        self.hexface[0].node[i1-1] = inode

                    if j == self.nglly-1:
                        i3 += 1
                        self.hexface[2].node[i3-1] = inode

                    if k == 0:
                        i5 += 1
                        self.hexface[4].node[i5-1] = inode

                    if k == self.ngllz-1:
                        i6 += 1
                        self.hexface[5].node[i6-1] = inode


        for i_face in range(6):
            if   i_face == 0 or i_face == 2: # ZX PLANE
                self.hexface[i_face].gnode[0] = self.hexface[i_face].node[0]
                self.hexface[i_face].gnode[1] = self.hexface[i_face].node[self.ngllx-1]
                self.hexface[i_face].gnode[2] = self.hexface[i_face].node[self.ngllzx-1]
                self.hexface[i_face].gnode[3] = self.hexface[i_face].node[self.ngllzx - self.ngllx]
            elif i_face == 1 or i_face == 3: # YZ PLANE
                self.hexface[i_face].gnode[0] = self.hexface[i_face].node[0]
                self.hexface[i_face].gnode[1] = self.hexface[i_face].node[self.nglly - 1]
                self.hexface[i_face].gnode[2] = self.hexface[i_face].node[self.ngllyz - 1]
                self.hexface[i_face].gnode[3] = self.hexface[i_face].node[self.ngllyz - self.nglly]
            elif i_face == 4 or i_face == 5: # XY PLANE
                self.hexface[i_face].gnode[0] = self.hexface[i_face].node[0]
                self.hexface[i_face].gnode[1] = self.hexface[i_face].node[self.ngllx - 1]
                self.hexface[i_face].gnode[2] = self.hexface[i_face].node[self.ngllxy - 1]
                self.hexface[i_face].gnode[3] = self.hexface[i_face].node[self.ngllxy - self.ngllx]

        # Set the signs of the normals for each face:
        hexface_sign = [ 1.00000000,
                         1.00000000,
                        -1.00000000,
                        -1.00000000,
                        -1.00000000,
                         1.00000000]


        # Degrees of freedom on each face:

        # First we initialise the size of each EDOF array for each hexface:
        self.hexface[0].init_edof(self.nndof * self.ngllzx)
        self.hexface[1].init_edof(self.nndof * self.ngllyz)
        self.hexface[2].init_edof(self.nndof * self.ngllzx)
        self.hexface[3].init_edof(self.nndof * self.ngllyz)
        self.hexface[4].init_edof(self.nndof * self.ngllxy)
        self.hexface[5].init_edof(self.nndof * self.ngllxy)

        ngll_per_face = {0: self.ngllzx,
                         1: self.ngllyz,
                         2: self.ngllzx,
                         3: self.ngllyz,
                         4: self.ngllxy,
                         5: self.ngllxy
                        }

        for i_face in range(6):
            idof1 = 0
            idof2 = self.nndof
            for i_gll in range(ngll_per_face[i_face]):
                self.hexface[i_face].edof[idof1:idof2] = (self.hexface[i_face].node[i_gll]-1)*self.nndof \
                                                         + np.arange(1,self.nndof+1)
                idof1 += copy(self.nndof)
                idof2 += copy(self.nndof)


        # Allocate the face edge node arrays:
        self.indx = np.zeros(self.ngllx, dtype=int)
        self.indy = np.zeros(self.nglly, dtype=int)
        self.indz = np.zeros(self.ngllz, dtype=int)

        # now initialise the hex_face_edge - this is a 6 by 4 array of
        # hexface_edge objects 6 hexfaces with 4 edges each :
        self.hexface_edge = [[hex_face_edge(), hex_face_edge(), hex_face_edge(), hex_face_edge()],
                             [hex_face_edge(), hex_face_edge(), hex_face_edge(), hex_face_edge()],
                             [hex_face_edge(), hex_face_edge(), hex_face_edge(), hex_face_edge()],
                             [hex_face_edge(), hex_face_edge(), hex_face_edge(), hex_face_edge()],
                             [hex_face_edge(), hex_face_edge(), hex_face_edge(), hex_face_edge()],
                             [hex_face_edge(), hex_face_edge(), hex_face_edge(), hex_face_edge()],
                             ]


        # Face 1:
        self._set_face(iface=int(1),
                       nx = copy(self.ngllx),
                       ny=copy(self.ngllz),
                       indarr1=self.indx,
                       indarr2=self.indy)
        # Face 2:
        self._set_face(iface=int(2),
                       nx=copy(self.nglly),
                       ny=copy(self.ngllz),
                       indarr1=self.indy,
                       indarr2=self.indz)

        # Face 3:
        self._set_face(iface=int(3),
                       nx=copy(self.ngllx),
                       ny=copy(self.ngllz),
                       indarr1=self.indx,
                       indarr2=self.indz)

        # Face 4:
        self._set_face(iface=int(4),
                       nx=copy(self.nglly),
                       ny=copy(self.ngllz),
                       indarr1=self.indy,
                       indarr2=self.indz)

        # Face 5:
        self._set_face(iface=int(5),
                       nx=copy(self.ngllx),
                       ny=copy(self.nglly),
                       indarr1=self.indx,
                       indarr2=self.indy)

        # Face 6:
        self._set_face(iface=int(6),
                       nx=copy(self.ngllx),
                       ny=copy(self.nglly),
                       indarr1=self.indx,
                       indarr2=self.indy)

        print("Completed prepare_hexface.")

    def _set_face(self, iface, nx, ny, indarr1, indarr2):
        #edge 1
        self._edge_loop(j_low=0, j_high=1, i_low=0, i_high=nx,
                        iedge=0, iface=iface, ind_arr=indarr1, nx=nx,
                        loop_ij='i')
        # edge 2
        self._edge_loop(j_low=0, j_high=ny, i_low=nx-1, i_high=nx,
                        iedge=1, iface=iface, ind_arr=indarr2, nx=nx,
                        loop_ij='j')
        # edge 3

        self._edge_loop(j_low=ny-1, j_high=ny, i_low=0, i_high=nx,
                        iedge=2, iface=iface, ind_arr=indarr1, nx=nx,
                        loop_ij='i')
        # edge 4
        self._edge_loop(j_low=0, j_high=ny, i_low=0, i_high=1,
                        iedge=3, iface=iface, ind_arr=indarr2, nx=nx,
                        loop_ij='j')





    def prepare_integration(self):

        # Allocate some arrays:
        self.dshape_hex8   = np.zeros((self.ndim, self.ngnode, self.ngll))
        self.gll_weights   = np.zeros(self.ngll)
        self.lagrange_gll  = np.zeros((self.ngll, self.ngll))
        self.dlagrange_gll = np.zeros((self.ndim, self.ngll, self.ngll))

        # Call dshape_function_hex8 function
        self._dshape_function_hex8()

        # Call gll_quadrature function
        self._gll_quadrature()


    def _dshape_function_hex8(self):
        # Computes derivatives of the shape functions at GLL points

        # initialise:
        self.dshape_hex8 = np.zeros((3,self.ngnod,self.ngllx*self.nglly*self.ngllz))

        one = 1.0000000
        one_eighth = 0.12500000


        print("NOTE WE ARE USING GLLPZ WHEN THEY USE ZETAGLL ETC")
        igll = -1
        for k in range(self.ngllz):
            zetap = one + self.gllpz[k]
            zetam = one - self.gllpz[k]

            for j in range(self.nglly):
                etap = one + self.gllpy[j]
                etam = one - self.gllpy[j]

                for i in range(self.ngllx):
                    igll += 1
                    xip = one + self.gllpx[i]
                    xim = one - self.gllpx[i]

                    self.dshape_hex8[0, 0, igll] = - one_eighth * etam * zetam
                    self.dshape_hex8[0, 1, igll] = one_eighth * etam * zetam
                    self.dshape_hex8[0, 2, igll] = one_eighth * etap * zetam
                    self.dshape_hex8[0, 3, igll] = - one_eighth * etap * zetam
                    self.dshape_hex8[0, 4, igll] = - one_eighth * etam * zetap
                    self.dshape_hex8[0, 5, igll] = one_eighth * etam * zetap
                    self.dshape_hex8[0, 6, igll] = one_eighth * etap * zetap
                    self.dshape_hex8[0, 7, igll] = - one_eighth * etap * zetap

                    self.dshape_hex8[1, 0, igll] = - one_eighth * xim * zetam
                    self.dshape_hex8[1, 1, igll] = - one_eighth * xip * zetam
                    self.dshape_hex8[1, 2, igll] = one_eighth * xip * zetam
                    self.dshape_hex8[1, 3, igll] = one_eighth * xim * zetam
                    self.dshape_hex8[1, 4, igll] = - one_eighth * xim * zetap
                    self.dshape_hex8[1, 5, igll] = - one_eighth * xip * zetap
                    self.dshape_hex8[1, 6, igll] = one_eighth * xip * zetap
                    self.dshape_hex8[1, 7, igll] = one_eighth * xim * zetap

                    self.dshape_hex8[2, 0, igll] = - one_eighth * xim * etam
                    self.dshape_hex8[2, 1, igll] = - one_eighth * xip * etam
                    self.dshape_hex8[2, 2, igll] = - one_eighth * xip * etap
                    self.dshape_hex8[2, 3, igll] = - one_eighth * xim * etap
                    self.dshape_hex8[2, 4, igll] = one_eighth * xim * etam
                    self.dshape_hex8[2, 5, igll] = one_eighth * xip * etam
                    self.dshape_hex8[2, 6, igll] = one_eighth * xip * etap
                    self.dshape_hex8[2, 7, igll] = one_eighth * xim * etap




    def _gll_quadrature(self):

        gll_weights   = np.zeros(self.ngll)
        lagrange_gll  = np.zeros((self.ngll,self.ngll))
        dlagrange_gll = np.zeros((self.ndim, self.ngll, self.ngll))
        gll_points    = np.zeros((self.ndim, self.ngll))
        lagrange_x    = np.zeros(self.ngllx)
        lagrange_y    = np.zeros(self.nglly)
        lagrange_z    = np.zeros(self.ngllz)


        dlx = lagrange1st(self.ngllx - 1)


        # Sorting coordinates of GLL points:
        n = -1
        for k in range(self.ngllz):
            for j in range(self.nglly):
                for i in range(self.ngllx):
                    n+=1
                    # points of integration
                    gll_points[0, n] = self.gllpx[i]
                    gll_points[1, n] = self.gllpy[j]
                    gll_points[2, n] = self.gllpz[k]

                    gll_weights[n] = self.gllwx[i] * self.gllwy[j] * self.gllwz[k]


        i_ind = {0: [0, 0, 0],
                 1: [1, 0, 0],
                 2: [2, 0, 0],
                 }


        # need to make an indexing array:
        we_ind = np.zeros((self.ngll, 3), dtype=int)
        wectr = 0
        for k in range(self.ngllz):
            for j in range(self.nglly):
                for i in range(self.ngllx):
                    we_ind[wectr, 0] = i
                    we_ind[wectr, 1] = j
                    we_ind[wectr, 2] = k
                    wectr+= 1

        for ii in range(self.ngll):
            xi   = gll_points[0,ii]
            eta  = gll_points[1,ii]
            zeta = gll_points[2,ii]

            # Calculating lagrange values at each point
            for i in range(self.ngllx):
                lagrange_x[i] = lagrange(N=self.ngllx-1, i=i-1, x=xi)
            for i in range(self.nglly):
                lagrange_y[i] = lagrange(N=self.nglly-1, i=i-1, x=eta)
            for i in range(self.ngllz):
                lagrange_z[i] = lagrange(N=self.ngllz - 1, i=i - 1, x=zeta)

            # Calculating derivative lagrange values at each point
            # Also first line saves the lagrange_gll value as the product
            # of the above arrays:






            n = -1
            for k in range(self.ngllz):
                for j in range(self.nglly):
                    for i in range(self.ngllx):
                        n+=1

                        # Store lagrange value:
                        lagrange_gll[ii, n] = lagrange_x[i] * lagrange_y[j] * lagrange_z[k]

                        dlagrange_gll[0,ii,n] = dlx[:, we_ind[ii,:][0]][i] * lagrange_y[j]  * lagrange_z[k]
                        dlagrange_gll[1,ii,n] = lagrange_x[i] * dlx[:, we_ind[ii,:][1]][j]  * lagrange_z[k]
                        dlagrange_gll[2,ii,n] = lagrange_x[i] * lagrange_y[j] * dlx[:, we_ind[ii,:][2]][k]


        # store relevant parameters:
        self.dlagrange_gll  = dlagrange_gll
        self.gll_weights    = gll_weights
        self.lagrange_gll   = lagrange_gll
        self.gll_points     = gll_points
        self.lagrange_x     = lagrange_x
        self.lagrange_y     = lagrange_y
        self.lagrange_z     = lagrange_z
