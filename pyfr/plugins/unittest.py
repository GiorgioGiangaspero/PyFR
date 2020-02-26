# -*- coding: utf-8 -*-

from collections import defaultdict

import numpy as np

from pyfr.mpiutil import get_comm_rank_root, get_mpi
from pyfr.plugins.base import BasePlugin, init_csv

import re

class UnitTestPlugin(BasePlugin):
	name = 'unittest'    
	systems = ['ac-euler', 'ac-navier-stokes', 'euler', 'navier-stokes']
	formulations = ['dual', 'std']

	def __init__(self, intg, cfgsect, suffix):
		super().__init__(intg, cfgsect, suffix)

		comm, rank, root = get_comm_rank_root()

		bcname = self.cfg.getliteral(cfgsect, 'bcname')
		bc = re.match(r'periodic[ _-]([a-z0-9]+)[ _-](l|r)$', bcname)

		if bc.group(2) == 'l':
			flg = int(bc.group(1)) + 1
			print('l group flag = {}'.format(flg))
		elif bc.group(2) == 'r':
			flg = -(int(bc.group(1)) + 1)
			print('r group flag = {}'.format(flg))
		else:
			raise ValueError('Cannot find periodic boundary.')

		# Underlying elements class
		self.elementscls = intg.system.elementscls

		# Get the mesh and elements
		mesh, elemap = intg.system.mesh, intg.system.ele_map

		# Interpolation matrices and quadrature weights
		self._m0 = m0 = {}
		self._qwts = qwts = defaultdict(list)

		con = mesh['con_p{0}'.format(intg.rallocs.prank)]

		self.haspbc = False
		pbc = []
		intg.system.area = 0.0

		for face in con:
			for elem in face:
				if elem[-1] == flg:
					pbc.append(elem)

		if len(pbc) != 0:
			self.haspbc = True
			# Element indices and associated face normals
			eidxs = defaultdict(list)
			norms = defaultdict(list)

			for i in range(len(pbc)):
				etype, eidx, fidx, flags = pbc[i].astype('U4,i4,i1,i1')
				eles = elemap[etype]

				if (etype, fidx) not in m0:
					facefpts = eles.basis.facefpts[fidx]

					m0[etype, fidx] = eles.basis.m0[facefpts]
					qwts[etype, fidx] = eles.basis.fpts_wts[facefpts]

				# Unit physical normals and their magnitudes (including |J|)
				npn = eles.get_norm_pnorms(eidx, fidx)
				mpn = eles.get_mag_pnorms(eidx, fidx)

				eidxs[etype, fidx].append(eidx)
				norms[etype, fidx].append(mpn[:, None]*npn)

			self._eidxs = {k: np.array(v) for k, v in eidxs.items()}
			self._norms = {k: np.array(v) for k, v in norms.items()}


		if not self.haspbc:
			intg.system.area = comm.allreduce(0.0, op=get_mpi('sum')) 

		else:
			# Solution matrices indexed by element type
			solns = dict(zip(intg.system.ele_types, intg.soln))
			ndims, nvars = self.ndims, self.nvars

			area = np.zeros(ndims)

			for etype, fidx in self._m0:
				# Get the interpolation operator
				m0 = self._m0[etype, fidx]
				nfpts, nupts = m0.shape

				# Extract the relevant elements from the solution
				uupts = solns[etype][..., self._eidxs[etype, fidx]]

				# Interpolate to the face
				ufpts = np.dot(m0, uupts.reshape(nupts, -1))
				ufpts = ufpts.reshape(nfpts, nvars, -1)
				ufpts = ufpts.swapaxes(0, 1)

				sol = self.elementscls.con_to_pri(ufpts, self.cfg)[0]
				ones = np.ones(np.shape(sol))

				# Get the quadrature weights and normal vectors
				qwts = self._qwts[etype, fidx]
				norms = self._norms[etype, fidx]

				# Do the quadrature
				area[:ndims] += np.einsum('i...,ij,jik', qwts, ones, norms)

			intg.system.area= comm.allreduce(area, op=get_mpi('sum')) 

		if rank == root:
			print('(Area)*(Face normal) for', bcname, ': ', np.sum(intg.system.area))

		raise ValueError('End unit test.')
