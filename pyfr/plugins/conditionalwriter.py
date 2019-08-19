# -*- coding: utf-8 -*-

from collections import defaultdict

import numpy as np

from pyfr.inifile import Inifile
from pyfr.mpiutil import get_comm_rank_root, get_mpi
from pyfr.plugins.base import BasePlugin
from pyfr.writers.native import NativeWriter


class ConditionalWriterPlugin(BasePlugin):
    name = 'conditionalwriter'
    systems = ['ac-euler', 'ac-navier-stokes', 'euler', 'navier-stokes']
    formulations = ['dual', 'std']

    def __init__(self, intg, cfgsect, suffix):
        super().__init__(intg, cfgsect, suffix)

        comm, rank, root = get_comm_rank_root()

        # Construct the solution writer
        basedir = self.cfg.getpath(cfgsect, 'basedir', '.', abs=True)
        basename = self.cfg.get(cfgsect, 'basename')
        self._writer = NativeWriter(intg, self.nvars, basedir, basename,
                                    prefix='soln')

        # Output field names
        self.fields = intg.system.elementscls.convarmap[self.ndims]

        # Threshold value of pressure on the boundary under which the solution
        # is written to file
        self.p_threshold = self.cfg.getfloat(cfgsect, 'p-threshold')

        # Boundaries of the box to be monitored.
        self.bounds = self.cfg.getliteral(cfgsect, 'bounds')

        # Frequency of the check for the condition
        self.nsteps = self.cfg.getint(cfgsect, 'nsteps')

        # Check if the system is incompressible
        self._ac = intg.system.name.startswith('ac')

        # Underlying elements class
        self.elementscls = intg.system.elementscls

        # Boundary to monitor
        bc = 'bcon_{0}_p{1}'.format(suffix, intg.rallocs.prank)

        # Get the mesh and elements
        mesh, elemap = intg.system.mesh, intg.system.ele_map

        # See which ranks have the boundary
        bcranks = comm.gather(bc in mesh, root=root)

        # The root rank checks the existence of the bc in the mesh
        if rank == root:
            if not any(bcranks):
                raise RuntimeError('Boundary {0} does not exist'
                                   .format(suffix))

        # Interpolation matrices
        self._m0 = m0 = {}

        # If we have the boundary then process the interface
        if bc in mesh:
            # Element indices
            eidxs = defaultdict(list)

            for etype, eidx, fidx, flags in mesh[bc].astype('U4,i4,i1,i1'):
                eles = elemap[etype]

                # skip this element if the coordinates of its flux points are
                # outside the bounds
                plocfpts = eles.get_ploc_for_inter(eidx, fidx)                
                if not self.el_is_within_bounds(plocfpts):
                    continue

                if (etype, fidx) not in m0:
                    facefpts = eles.basis.facefpts[fidx]

                    m0[etype, fidx] = eles.basis.m0[facefpts]

                eidxs[etype, fidx].append(eidx)

            self._eidxs = {k: np.array(v) for k, v in eidxs.items()}

    
    def el_is_within_bounds(self, plocfpts):
        for dim in range(len(self.bounds)):
            minb = self.bounds[dim][0]
            maxb = self.bounds[dim][1]
            if np.any(plocfpts[:,dim] < minb) or np.any(plocfpts[:,dim] > maxb):
                #print('plocfpts = {}'.format(plocfpts))
                #print('minb = {}, maxb = {}'.format(minb, maxb))
                #print('Skipping ploc {}. Dim  = {}'.format(plocfpts[:,dim],dim))
                return False
        return True


    def __call__(self, intg):
        # Return if no output is due
        if intg.nacptsteps % self.nsteps:
            return

        # MPI info
        comm, rank, root = get_comm_rank_root()

        # Solution matrices indexed by element type
        solns = dict(zip(intg.system.ele_types, intg.soln))
        nvars = self.nvars

        # initialize pmin to an absurd high value
        pmin = np.array(1e10,'d')

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

            # Compute the pressure
            pidx = 0 if self._ac else -1
            p = self.elementscls.con_to_pri(ufpts, self.cfg)[pidx]

            np.minimum(pmin, p.min(), out=pmin)

        #Everybody gets the minimum across all processes
        comm.Allreduce(get_mpi('in_place'), pmin, op=get_mpi('min'))

        # check whether we need to write the solution or not
        if pmin <= self.p_threshold:

            stats = Inifile()
            stats.set('data', 'fields', ','.join(self.fields))
            stats.set('data', 'prefix', 'soln')
            intg.collect_stats(stats)

            # Prepare the metadata
            metadata = dict(intg.cfgmeta,
                            stats=stats.tostr(),
                            mesh_uuid=intg.mesh_uuid)

            # Write out the file
            solnfname = self._writer.write(intg.soln, metadata, intg.tcurr)

            # If a post-action has been registered then invoke it
            self._invoke_postaction(mesh=intg.system.mesh.fname, soln=solnfname,
                                    t=intg.tcurr)
