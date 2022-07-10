# Class for the mesh
import csv
from copy import copy
import numpy as np
from GLL import *
from hex_classes import hex_face, hex_face_edge


g_acc      = 9.8200000000 # I think they use this instead of 9.81
GRAV_CONS  = 6.67408e-11

class Material_Block():
    def __init__(self):
        pass

    def init_elmt(self, dim):
        self.elmt = np.zeros((dim), dtype=int)

    def set_elmt(self, val):
        self.elmt = val


class Mesh():
    def __init__(self, path, fname, ngllx, nglly, ngllz, ngnode,
                 ISDISP_DOF=True, ISPOT_DOF=True):
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

        self.maxngll2d = copy(self.ngllxy)  # NEEDS PROPER IMPLEMENTATION
                                            # if ngllx != nglly or ngllz

        self.infbc = False
        self.nondimensionalise = True

        self.ISDISP_DOF = ISDISP_DOF
        self.ISPOT_DOF = ISPOT_DOF

        self.nenode = copy(self.ngll) # This is the number of element nodes
                                      # where as self.nnode is the number of unique
                                      # nodes




        # Initialise arrays:
        self.massden_elmt = np.zeros((self.ngll, self.nelem))
        self.shearmod_elmt = np.zeros((self.ngll, self.nelem))
        self.bulkmod_elmt = np.zeros((self.ngll, self.nelem))



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
                        self.nmatblk = int(row[0])
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



    def initialise_dof(self):
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
        if self.ISDISP_DOF:
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

        if self.ISPOT_DOF:
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



    def set_element_dof(self):
        # Sets IDs for the element level DOF for U and Phi so we can map the element
        # matrices

        self.edofu[:] = -9999

        if self.ISPOT_DOF:
            self.edofphi[:] = -9999

        iu0   = 0
        iphi0 = 0
        iphi = 0
        nu = 0
        iu = np.zeros(self.nndofu, dtype=int) # SOME KIND OF ARRAY

        for i in range(self.ngll):
            if self.ISDISP_DOF:
                iu[0] = iu0 + 1
                nu += 1
                self.edofu[nu-1] = iu[0]

                for j in range(1, self.nndofu):

                    nu += 1

                    iu[j] = iu[j-1] + 1
                    self.edofu[nu-1] = iu[j]

                iu0 = iu[self.nndofu-1]
                iphi0 = iu[self.nndofu-1]


            if self.ISPOT_DOF:
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

                    if np.abs(sum_shape)-1 > 1e-10:
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




    def prepare_integration2d(self):

        # allocate:
        #   Derivatives of shape functions (3D):
        self.dshape_quad4_xy = np.zeros((2, 4, self.ngllxy))
        self.dshape_quad4_yz = np.zeros((2, 4, self.ngllyz))
        self.dshape_quad4_zx = np.zeros((2, 4, self.ngllzx))

        #   XY stuff:
        self.gll_weights_xy, self.gll_points_xy, self.lagrange_gll_xy, self.dlagrange_gll_xy = \
            self._allocator_gll_factory(self.ngllxy)
        #   YZ stuff:
        self.gll_weights_yz, self.gll_points_yz, self.lagrange_gll_yz, self.dlagrange_gll_yz = \
            self._allocator_gll_factory(self.ngllyz)
        #   ZX stuff:
        self.gll_weights_zx, self.gll_points_zx, self.lagrange_gll_zx, self.dlagrange_gll_zx = \
            self._allocator_gll_factory(self.ngllzx)


        # Create the derivative shape functions QUAD4 for XY, YZ, ZX
        self._dshape_function_quad4_factory('xy')
        self._dshape_function_quad4_factory('yz')
        self._dshape_function_quad4_factory('zx')

        # Now calculate 2D quadrature for each direction:
        #   XY:
        self._gll_quadrature2d(self.ngllx, self.nglly, self.ngllxy, self.gll_points_xy, self.gll_weights_xy,
                               self.lagrange_gll_xy, self.dlagrange_gll_xy)
        #   YZ:
        self._gll_quadrature2d(self.nglly, self.ngllz, self.ngllyz, self.gll_points_yz, self.gll_weights_yz,
                               self.lagrange_gll_yz, self.dlagrange_gll_yz)
        #   ZX:
        self._gll_quadrature2d(self.ngllz, self.ngllx, self.ngllzx, self.gll_points_zx, self.gll_weights_zx,
                               self.lagrange_gll_zx, self.dlagrange_gll_zx)

        print("Finished preparing integration 2D.")



    def _gll_quadrature2d(self, ngllx, nglly, ngll, gll_points2d, gll_weights2d, lagrange_gll2d, dlagrange_gll2d):
        # Allocate:
        lagrange_x = np.zeros(self.ngllx)
        lagrange_y = np.zeros(self.nglly)

        dlx = lagrange1st(self.ngllx - 1)


        n = -1
        for j in range(nglly):
            for i in range(ngllx):
                n += 1

                gll_points2d[0, n] = self.gllpx[i]
                gll_points2d[1, n] = self.gllpy[j]

                gll_weights2d[n]   = self.gllwx[i] * self.gllwy[j]


        # need to make an indexing array for use below:
        we_ind = np.zeros((self.ngll, 2), dtype=int)
        wectr = 0
        for j in range(self.nglly):
            for i in range(self.ngllx):
                we_ind[wectr, 0] = i
                we_ind[wectr, 1] = j
                wectr += 1


        # Get the 2D lagrange and derivatives on face:
        for ii in range(ngll):
            xi  = gll_points2d[0,ii]
            eta = gll_points2d[1,ii]


            # Calculating lagrange values at each point
            for i in range(self.ngllx):
                lagrange_x[i] = lagrange(N=self.ngllx-1, i=i-1, x=xi)
            for i in range(self.nglly):
                lagrange_y[i] = lagrange(N=self.nglly-1, i=i-1, x=eta)


            # Calculating derivative lagrange values at each point
            # Also first line saves the lagrange_gll value as the product
            # of the above arrays:
            n = -1
            for j in range(self.nglly):
                for i in range(self.ngllx):
                    n+=1

                    # Store lagrange value:
                    lagrange_gll2d[ii, n] = lagrange_x[i] * lagrange_y[j]

                    dlagrange_gll2d[0,ii,n] = dlx[:, we_ind[ii,:][0]][i] * lagrange_y[j]
                    dlagrange_gll2d[1,ii,n] = lagrange_x[i] * dlx[:, we_ind[ii,:][1]][j]



    def _dshape_function_quad4_factory(self, plane):
        # Determine which plane:
        if plane == 'xy':
             ngllx         = copy(self.ngllx)
             nglly         = copy(self.ngllx)
             xigll         = copy(self.gllpx)
             etagll        = copy(self.gllpy)
             dshape_quad4  = self.dshape_quad4_xy
        elif plane == 'yz':
             ngllx         = copy(self.nglly)
             nglly         = copy(self.ngllz)
             xigll         = copy(self.gllpy)
             etagll        = copy(self.gllpz)
             dshape_quad4  = self.dshape_quad4_yz
        elif plane == 'zx':
             ngllx         = copy(self.ngllz)
             nglly         = copy(self.ngllx)
             xigll         = copy(self.gllpz)
             etagll        = copy(self.gllpx)
             dshape_quad4  = self.dshape_quad4_zx
        else:
            raise ValueError("Must be xy, yz, zx")


        # Now we can call to the actual function:
        self._dshape_function_quad4(ngllx, nglly, xigll, etagll, dshape_quad4)




    def _dshape_function_quad4(self, ngllx, nglly, xigll, etagll, dshape_quad4):
        one = 1.0000000
        one_fourth = 0.25000000
        local_ngll = ngllx * nglly

        # Compute derivatives of 2D shape functions:
        igll = -1
        for j in range(nglly):
            etap = one + etagll[j]
            etam = one - etagll[j]
            for i in range(ngllx):
                igll += 1
                xip = one + xigll[i]
                xim = one - xigll[i]
                dshape_quad4[0,0,igll] = -one_fourth*etam
                dshape_quad4[0,1,igll] = one_fourth*etam
                dshape_quad4[0,2,igll] = one_fourth*etap
                dshape_quad4[0,3,igll] = -one_fourth*etap
                dshape_quad4[1,0,igll] = -one_fourth*xim
                dshape_quad4[1,1,igll] = -one_fourth*xip
                dshape_quad4[1,2,igll] = one_fourth*xip
                dshape_quad4[1,3,igll] = one_fourth*xim



    def _allocator_gll_factory(self, d):
        gll_weights    = np.zeros(d)
        gll_points     = np.zeros((2, d))
        lagrange_gll   = np.zeros((d, d))
        d_lagrange_gll = np.zeros((2, d, d))
        return gll_weights, gll_points, lagrange_gll, d_lagrange_gll



    def set_model_properties(self):
        print('Set_model_properties currently does not support parallel (ghosts) or infinite elements')
        # allocate
        ielmts = np.zeros(self.nmatblk, dtype=int)
        num = np.zeros(self.ngll)
        block_nelmt = np.zeros(self.nmatblk, dtype=int)

        # loop through each material block:
        for i_blk in range(self.nmatblk):
            block_nelmt[i_blk] = len(self.matID[self.matID == i_blk+1])

        # Now create a list of length nmatblok  with a material_block object in each element
        block = []
        for j in range(self.nmatblk):
            block.append(Material_Block())
            # Now allocate the elmt of each Material Block object in the listL
            block[j].init_elmt(dim=block_nelmt[j])


        for i in range(self.nelem):
            iblk = self.matID[i]
            ielmts[iblk-1] += 1
            block[iblk-1].elmt[ielmts[iblk-1]-1] = i

        # Now each of the Material_block objects in 'block' has its elmt array
        # defined - these arrays hold the IDs of each spectral element that
        # belong to that material. E.g. if we have 2 materials with 4 elements of material 1
        # and 3 of material 2 then block is a list with 2 MaterialBlock objects. The first of these
        # objects has an elmt of length 4 and the second has an elmt of length 2 where the values in those
        # arrays are the element IDs.

        # Now convert block model to pointwise model
        self._convert_block_to_point_model(block)


    def _convert_block_to_point_model(self, block, isdensity=True):
        # Need to consider each material that is defined which is in the matfile with the following:
        # Simplified version of original code where I have just done the bits I need.
        # materialID, domainID, type, γ, E, ν, φ, c, ψ
        #   domainID  - elastic = 1 or viscoelastic = 11
        #   type      - material structure -  0 = homogenous
        #                                  - -1 = tomographic

        # Currently only supporting homogenous for which the 6 params above are defined:
        #       γ   - weight [kN/m^3]              E   - Youngs Modulus [kN/m^2]
        #       ν   - poissons?                    φ   - angle of internal friction [deg]
        #       c   - cohesion [kN/m^2]            ψ   - angle of dilation [deg]

        self.mat_domain = np.zeros(self.nmatblk, dtype=int)
        self.type_blk = np.zeros(self.nmatblk, dtype=int)
        self.bulkmod_blk = np.zeros(self.nmatblk)
        self.shearmod_blk = np.zeros(self.nmatblk)
        self.rho_blk = np.zeros(self.nmatblk)
        self.gam_blk = np.zeros(self.nmatblk)
        self.ym_blk = np.zeros(self.nmatblk)
        self.nu_blk = np.zeros(self.nmatblk)
        self.phi_blk = np.zeros(self.nmatblk)
        self.coh_blk = np.zeros(self.nmatblk)
        self.psi_blk = np.zeros(self.nmatblk)
        self.isempty_blk = np.zeros(self.nmatblk, dtype=bool)
        self.ismat = np.zeros(self.nmatblk, dtype=bool) # All false

        # For each material
        for i_blk in range(self.nmatblk):
            material = self.materials[i_blk]
            imat       = int(material[0]) - 1 # for py indexing

            self.mat_domain[imat] = int(material[1])
            self.type_blk[imat]   = int(material[2])

            if self.type_blk[imat]  == 0 :
                # Block model (homogenous)
                # Defining density based params.
                if isdensity:
                    self.rho_blk[imat] = float(material[3])
                    self.gam_blk[imat] = self.rho_blk[imat]*g_acc
                else:
                    self.gam_blk[imat] = float(material[3])
                    self.rho_blk[imat] = self.gam_blk[imat]/g_acc

            self.ym_blk[imat] = float(material[4])
            self.nu_blk[imat] = float(material[5])
            self.phi_blk[imat] = float(material[6])
            self.coh_blk[imat] = float(material[7])
            self.psi_blk[imat] = float(material[8])


            # Calculate shear/bulk modulus:
            self.bulkmod_blk[imat] = self.ym_blk[imat]/(3*(1 - 2*self.nu_blk[imat]))
            self.shearmod_blk[imat] = 0.5 * self.ym_blk[imat]/(1 + self.nu_blk[imat])



            # Checking if 0 values:
            if np.logical_and(self.rho_blk[imat]==0, self.ym_blk[imat]==0):
                self.isempty_blk[imat] = True

            self.ismat[imat] = True


            # Homogenous material
            if self.ISDISP_DOF:
                self.bulkmod_elmt[:, list(block[i_blk].elmt) ]  = self.bulkmod_blk[imat]
                self.shearmod_elmt[:, list(block[i_blk].elmt) ] = self.shearmod_blk[imat]

            # need density if doing gravity or disp.
            if self.ISDISP_DOF or self.ISPOT_DOF:
                self.massden_elmt[:, list(block[i_blk].elmt)] = self.rho_blk[imat]


        print("NEED TO TEST THAT CALCULATED BULK/SHEAR/DENSITY are correct.")




    def set_nondimensional_params(self):

        # DENSITY
        if self.infbc:
            raise ValueError('Not implemented currently. ')
        else:
            self.mindensity = np.min(self.massden_elmt.flatten())
            self.maxdensity = np.max(self.massden_elmt.flatten())
        # Always use positive value for nondimensionalizing
        self.maxdensity = np.max(np.array([np.abs(self.mindensity), np.abs(self.maxdensity)]))
        #print("Minimum density: ", self.mindensity)
        #print("Maximum density: ", self.maxdensity)

        # BULK MODULUS
        if self.infbc:
            raise ValueError('Not implemented currently. ')
        else:
            self.minbulkmod = np.min(self.bulkmod_elmt.flatten())
            self.maxbulkmod = np.max(self.bulkmod_elmt.flatten())
        # Always use positive value for nondimensionalizing
        self.maxbulkmod = np.max(np.array([np.abs(self.minbulkmod), np.abs(self.maxbulkmod)]))
        #print("Minimum kappa:   ", self.minbulkmod)
        #print("Maximum kappa:   ", self.maxbulkmod)

        # SHEAR MODULUS
        if self.infbc:
            raise ValueError('Not implemented currently. ')
        else:
            self.minshearmod = np.min(self.shearmod_elmt.flatten())
            self.maxshearmod = np.max(self.shearmod_elmt.flatten())
        # Always use positive value for nondimensionalizing
        self.maxshearmod = np.max(np.array([np.abs(self.minshearmod), np.abs(self.maxshearmod)]))
        #print("Minimum shear:   ", self.minshearmod)
        #print("Maximum shear:   ", self.maxshearmod)


    def calc_nondimensionalisation_vals(self):

        if self.nondimensionalise:
            # density
            self.dim_density     = copy(self.maxdensity)
            self.nondim_density  = 1.0 / self.dim_density

            # length scale (coords)
            self.dim_L           = self.absmaxcoord
            self.nondim_L        = 1.0 / self.dim_L

            # time? traction?
            self.nondim_T         = np.sqrt(np.pi * GRAV_CONS * self.dim_density)

            # velocity/acceleration
            self.dim_vel          = self.dim_L * self.nondim_T
            self.dim_accel        = self.dim_vel * self.nondim_T
            self.nondim_accel     = 1.0/self.dim_vel
            self.dim_m            = self.maxdensity * self.dim_L * self.dim_L * self.dim_L

            self.dim_mod          = self.dim_m * self.nondim_L * self.nondim_T * self.nondim_T
            self.nondim_mod       = 1.0/self.dim_mod

            self.dim_mtens        = self.dim_density * (self.dim_L**5) * self.nondim_T * self.nondim_T
            self.nondim_mtens     = 1.0/self.dim_mtens

            self.dim_gpot         = np.pi * GRAV_CONS * self.maxdensity * self.dim_L * self.dim_L
            self.dim_G            = np.pi * GRAV_CONS * self.maxdensity * self.dim_L
        else:
            # retain dimensional values
            self.dim_density         = 1.0
            self.nondim_density      = 1.0
            self.dim_L               = 1.0
            self.nondim_L            = 1.0
            self.nondim_T            = 1.0
            self.dim_vel             = 1.0
            self.dim_accel           = 1.0
            self.nondim_accel        = 1.0
            self.dim_m               = 1.0
            self.dim_mod             = 1.0
            self.nondim_mod          = 1.0
            self.dim_mtens           = 1.0
            self.nondim_mtens        = 1.0
            self.dim_gpot            = 1.0
            self.dim_G               = 1.0



    def calc_model_coord_extents(self):

        if self.infbc:
            raise ValueError("INFBC not implemented yet for calc_model_coords_extents")
        else:
            self.model_minx = np.min(self.g_coord[0,:])
            self.model_miny = np.min(self.g_coord[1,:])
            self.model_minz = np.min(self.g_coord[2,:])

            self.model_maxx = np.max(self.g_coord[0, :])
            self.model_maxy = np.max(self.g_coord[1, :])
            self.model_maxz = np.max(self.g_coord[2, :])

            self.mincoord   = np.min(np.array([self.model_minx, self.model_miny, self.model_minz]))
            self.maxcoord   = np.max(np.array([self.model_maxx, self.model_maxy, self.model_maxz]))

            self.absmaxx    = np.max(np.abs(self.g_coord[0,:]))
            self.absmaxy    = np.max(np.abs(self.g_coord[1,:]))
            self.absmaxz    = np.max(np.abs(self.g_coord[2,:]))
            self.absmaxcoord= np.max(np.array([self.absmaxx, self.absmaxy, self.absmaxz]))


    def apply_nondimensionalisation(self):
            self.g_coord = self.g_coord*self.nondim_L

            if self.ISDISP_DOF:
                self.massden_elmt  = self.massden_elmt  * self.nondim_density
                self.bulkmod_elmt  = self.bulkmod_elmt  * self.nondim_mod
                self.shearmod_elmt = self.shearmod_elmt * self.nondim_mod


    def read_input(self):
        # Load mesh stuff:
        self.g_num = np.loadtxt(f'{self.path}/{self.fname}_connectivity', skiprows=1, dtype=int)
        self._load_coords()
        self._load_free_surface()
        self._load_mat_IDs()
        self._load_mat_list()
        self._load_BCs()



    def prepare_free_surface(self):
        # Reads information from free surface file
        # must be called AFTER the spectral elements have been created

        # Read first line (number of elements in free surface set)
        self.nelmt_fs = int(self._read_desired_line( file=f'{self.path}/{self.fname}_free_surface', line=0)[0][0])

        # total poss number of nodes on free surface
        # we havent defined maxngll2d - pretty straight forward since this
        # currently only works when ngllx = nglly = ngllz
        nsnode_all = self.nelmt_fs * self.maxngll2d

        # Allocate arrays with correct dimension
        self.iface_fs    = np.zeros(self.nelmt_fs, dtype=int)
        self.gnum4_fs    = np.zeros((4, self.nelmt_fs), dtype=int)
        self.gnum_fs     = np.zeros((self.maxngll2d, self.nelmt_fs), dtype=int)
        self.nodelist    = np.zeros(nsnode_all, dtype=int)

        # Read all free surface values from file:
        fs_vals = np.loadtxt(f'{self.path}/{self.fname}_free_surface', skiprows=1).astype(int)

        self.iface_fs[:] = fs_vals[:,1]

        n1 = 1
        n2 = self.maxngll2d

        for i_face in range(self.nelmt_fs):
            num = copy(self.g_num[:, fs_vals[i_face, 0]-1])
            mask = list(self.hexface[fs_vals[i_face, 1]-1].gnode -1)
            self.gnum4_fs[:, i_face] = copy(num[mask])

            mask = list(self.hexface[fs_vals[i_face, 1] - 1].node - 1)
            self.gnum_fs[:, i_face] = copy(num[mask])


            self.nodelist[n1-1:n2] = copy(num[mask])
            n1 = n2 + 1
            n2 = n1 + self.maxngll2d - 1

        # Renumber Free Surface connectivity
        inode_order = self._i_uniinv(self.nodelist)




        self.nnode_fs = np.max(inode_order)

        isnode = np.zeros(self.nnode_fs, dtype=bool)

        # Store global node IDs for free surface nodes
        self.gnode_fs = np.zeros(self.nnode_fs, dtype=int)

        self.gnode_fs[inode_order[0]-1] = self.nodelist[0]

        isnode[inode_order[0]-1]        = True


        for i in range(1, nsnode_all):

            ind = inode_order[i]-1

            if not isnode[ind]:
                isnode[ind] = True
                self.gnode_fs[ind] = self.nodelist[i]

        n1 = 1
        n2 = self.maxngll2d
        self.rgnum_fs = np.zeros((self.maxngll2d, self.nelmt_fs))

        for i_face in range(self.nelmt_fs):
            self.rgnum_fs[:,i_face] = inode_order[n1-1:n2]

            n1 = n2 + 1
            n2 = n1 + self.maxngll2d - 1



    def compute_max_elementsize(self):
        # Must be called before converting HEX8 --> spectral elemenets
        maxsize = 0.

        for i_elmt in range(self.nelem):
            mdomain = self.mat_domain[self.matID[i_elmt]-1]

            # Doenst include transition.infinite elements
            if np.logical_or(mdomain == 1, mdomain == 11):
                num = self.g_num[self.hex8_gnode-1 , i_elmt]
                x1 = self.g_coord[:, num[0] - 1]
                x2 = self.g_coord[:, num[1] - 1]
                x3 = self.g_coord[:, num[2] - 1]
                x4 = self.g_coord[:, num[3] - 1]
                x5 = self.g_coord[:, num[4] - 1]
                x6 = self.g_coord[:, num[5] - 1]
                x7 = self.g_coord[:, num[6] - 1]
                x8 = self.g_coord[:, num[7] - 1]

                d1 = self._distance(x1, x7)
                d2 = self._distance(x2, x8)
                d3 = self._distance(x3, x5)
                d4 = self._distance(x4, x6)

                maxdiag = np.max(np.array([d1, d2, d3, d4]))

                # Replace max size if bigger
                if maxdiag > maxsize:
                    maxsize = copy(maxdiag)

            else:
                print('Skipping elements as are infinite or transitional.')

        self.maxsize_elmt = maxsize
        self.sqmaxsize_elmt = maxsize**2


    def _distance(self, a1, a2):
        assert(len(a1) == len(a2))

        a1 = np.array(a1)
        a2 = np.array(a2)

        adiff = np.square(a1-a2)
        return (np.sum(adiff))**0.5


    def determine_solver(self):
        print('We dont exactly have an abundance of solvers rn...!')

    def _i_uniinv(self, arr):
        # This function is a really long gross function in the math_library.f90
        # Essentially what it is doing is taking in an array and counting up
        # how many elements in the array are <= that element value
        # duplicate values are only counted once.

        len_arr = len(arr)
        new_arr = np.zeros(len_arr, dtype=int)

        # My alternative method:
        # 1) Create a seperate array with all the unique elements:
        uniq = np.unique(arr)

        # 2) Loop through original array values and get <= count from uniq:
        for i in range(len_arr):
            new_arr[i] = len(uniq[uniq<= arr[i] ])

        return new_arr

    def _read_desired_line(self, file, line):
        desired = [line]
        with open(file, 'r') as fin:
            reader = csv.reader(fin)
            out_line = [[s for s in row] for i, row in enumerate(reader) if i in desired]
        fin.close()
        return out_line

