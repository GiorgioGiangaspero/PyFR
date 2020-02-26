#!/usr/bin/env python
# -*- coding: utf-8 -*-

from argparse import ArgumentParser, FileType
import itertools as it
import os

import mpi4py.rc
mpi4py.rc.initialize = False

import h5py

from pyfr.backends import BaseBackend, get_backend
from pyfr.inifile import Inifile
from pyfr.mpiutil import register_finalize_handler
from pyfr.partitioners import BasePartitioner, get_partitioner
from pyfr.progress_bar import ProgressBar
from pyfr.rank_allocator import get_rank_allocation
from pyfr.readers import BaseReader, get_reader_by_name, get_reader_by_extn
from pyfr.readers.native import NativeReader
from pyfr.solvers import get_solver
from pyfr.util import subclasses
from pyfr.writers import BaseWriter, get_writer_by_name, get_writer_by_extn


def main():
    ap = ArgumentParser(prog='pyfr')
    sp = ap.add_subparsers(dest='cmd', help='sub-command help')

    # Common options
    ap.add_argument('--verbose', '-v', action='count')

    # Import command
    ap_import = sp.add_parser('import', help='import --help')
    ap_import.add_argument('inmesh', type=FileType('r'),
                           help='input mesh file')
    ap_import.add_argument('outmesh', help='output PyFR mesh file')
    types = sorted(cls.name for cls in subclasses(BaseReader))
    ap_import.add_argument('-t', dest='type', choices=types,
                           help='input file type; this is usually inferred '
                           'from the extension of inmesh')
    ap_import.set_defaults(process=process_import)

    # Partition command
    ap_partition = sp.add_parser('partition', help='partition --help')
    ap_partition.add_argument('np', help='number of partitions or a colon '
                              'delimited list of weights')
    ap_partition.add_argument('mesh', help='input mesh file')
    ap_partition.add_argument('solns', metavar='soln', nargs='*',
                              help='input solution files')
    ap_partition.add_argument('outd', help='output directory')
    partitioners = sorted(cls.name for cls in subclasses(BasePartitioner))
    ap_partition.add_argument('-p', dest='partitioner', choices=partitioners,
                              help='partitioner to use')
    ap_partition.add_argument('--popt', dest='popts', action='append',
                              default=[], metavar='key:value',
                              help='partitioner-specific option')
    ap_partition.add_argument('-t', dest='order', type=int, default=3,
                              help='target polynomial order; aids in '
                              'load-balancing mixed meshes')
    ap_partition.set_defaults(process=process_partition)

    # Export command
    ap_export = sp.add_parser('export', help='export --help')
    ap_export.add_argument('meshf', help='PyFR mesh file to be converted')
    ap_export.add_argument('solnf', help='PyFR solution file to be converted')
    ap_export.add_argument('outf', type=str, help='output file')
    types = [cls.name for cls in subclasses(BaseWriter)]
    ap_export.add_argument('-t', dest='type', choices=types, required=False,
                           help='output file type; this is usually inferred '
                           'from the extension of outf')
    ap_export.add_argument('-d', '--divisor', type=int, default=0,
                           help='sets the level to which high order elements '
                           'are divided; output is linear between nodes, so '
                           'increased resolution may be required')
    ap_export.add_argument('-g', '--gradients', action='store_true',
                           help='compute gradients')
    ap_export.add_argument('-p', '--precision', choices=['single', 'double'],
                           default='single', help='output number precision; '
                           'defaults to single')
    ap_export.set_defaults(process=process_export)

    #spanwise_avg command
    ap_spanwise_avg = sp.add_parser('spanwise_avg', help='spanwise_avg --help')
    ap_spanwise_avg.add_argument('inmesh', type=str, help='input mesh file')
    ap_spanwise_avg.add_argument('insolution', type=str,
                                 help='input solution file')
    ap_spanwise_avg.add_argument('spanwise_direction', choices=['x', 'y', 'z'],
                                 default='z', help='spanwise direction')
    ap_spanwise_avg.add_argument('streamwise_direction', choices=['x', 'y', 'z'],
                                 default='x', help='streamwise_direction direction')
    ap_spanwise_avg.add_argument('n_streamwise_stations', type=int,
                                 default=100, help='number of sample points along '
                                 'along the streamwise direction')
    ap_spanwise_avg.add_argument('n_spanwise_stations', type=int,
                                 default=100, help='number of sample points along '
                                 'along the spanwise direction')
    ap_spanwise_avg.add_argument('n_otherdir_stations', type=int,
                                 default=100, help='number of sample points along '
                                 'along the other direction (normal to streamwise '
                                 ' and spanwise')
    ap_spanwise_avg.add_argument('outsolution', type=str,
                                 default='spanwise_avg.csv', help='output .csv file')
    ap_spanwise_avg.set_defaults(process=process_spanwise_avg)

    # Run command
    ap_run = sp.add_parser('run', help='run --help')
    ap_run.add_argument('mesh', help='mesh file')
    ap_run.add_argument('cfg', type=FileType('r'), help='config file')
    ap_run.set_defaults(process=process_run)

    # Restart command
    ap_restart = sp.add_parser('restart', help='restart --help')
    ap_restart.add_argument('mesh', help='mesh file')
    ap_restart.add_argument('soln', help='solution file')
    ap_restart.add_argument('cfg', nargs='?', type=FileType('r'),
                            help='new config file')
    ap_restart.set_defaults(process=process_restart)

    # Options common to run and restart
    backends = sorted(cls.name for cls in subclasses(BaseBackend))
    for p in [ap_run, ap_restart]:
        p.add_argument('--backend', '-b', choices=backends, required=True,
                       help='backend to use')
        p.add_argument('--progress', '-p', action='store_true',
                       help='show a progress bar')

    # Parse the arguments
    args = ap.parse_args()

    # Invoke the process method
    if hasattr(args, 'process'):
        args.process(args)
    else:
        ap.print_help()

def get_eles(mesh, soln, cfg):
    from pyfr.shapes import BaseShape
    from pyfr.solvers.base import BaseSystem
    from pyfr.util import subclass_where, proxylist
    from collections import OrderedDict
    import re

    if soln:
        if mesh['mesh_uuid'] != soln['mesh_uuid']:
            raise RuntimeError('Mesh {} and solution {} have different '
                               'uuid'.format(mesh.fname, soln.fname))

    # Create a backend
    systemcls = subclass_where(BaseSystem,
                               name=cfg.get('solver', 'system'))

    # Get the elementscls
    elementscls = systemcls.elementscls

    basismap = {b.name: b for b in subclasses(BaseShape, just_leaf=True)}

    # Get the number of partitions
    ai  = mesh.array_info('spt')
    npr = max(int(re.search(r'\d+$', k).group(0)) for k in ai) + 1

    # Look for and load each element type from the mesh. One partition at the
    # time, save them in a list
    eles_partitions = []
    etypes_partitions = []
    for p in range(npr):
        elemap = OrderedDict()
        for f in mesh:
            m = re.match('spt_(.+?)_p{}$'.format(p), f)
            if m:
                # Element type
                t = m.group(1)

                elemap[t] = elementscls(basismap[t], mesh[f], cfg)

        # Process the solution
        if soln:
            prefix = Inifile(soln['stats']).get('data', 'prefix', 'soln')

            for k, ele in elemap.items():
                solnp = soln['{}_{}_p{}'.format(prefix, k, p)]
                ele.set_ics_from_soln(solnp, cfg)

        etypes_partitions.append(list(elemap.keys()))

        # Construct a proxylist to simplify collective operations
        eles_partitions.append(proxylist(elemap.values()))

    return eles_partitions, etypes_partitions

def process_spanwise_avg(args):
    from collections import OrderedDict
    from pyfr.plugins.sampler import _closest_upts
    import numpy as np

    try:
        from scipy.spatial import cKDTree
    except ImportError:
        raise RuntimeError('Process spanwise_avg requires the scipy package')


    # Read the input mesh
    in_mesh = NativeReader(args.inmesh)

    # Get the bounds of this mesh
    bounds = None # xyz, min/max
    for part in in_mesh.keys():
        if 'spt' in part:
            mmin = np.min(in_mesh[part].reshape(-1, in_mesh[part].shape[-1]), axis=0)
            mmax = np.max(in_mesh[part].reshape(-1, in_mesh[part].shape[-1]), axis=0)
            if bounds is not None:
                bounds[:, 0] = np.minimum(mmin, bounds[:,0])
                bounds[:, 1] = np.maximum(mmax, bounds[:,1])
            else:
                bounds = np.empty((3,2))
                bounds[:, 0] = mmin
                bounds[:, 1] = mmax

    # Read the in solution
    in_solution = NativeReader(args.insolution)

    # Read the in config from solution file
    in_cfg = Inifile(in_solution['config'])

    # Get the prefix and the number of variables stored in the input solution
    # The data file prefix defaults to soln for backwards compatibility
    stats_in = Inifile(in_solution['stats'])
    prefix = stats_in.get('data', 'prefix', 'soln')
    varnames = stats_in.get('data', 'fields').split(',')

    in_mesh_inf = in_mesh.array_info('spt')
    in_soln_inf = in_solution.array_info(prefix)

    ndims = next(iter(in_mesh_inf.values()))[1][2]
    nvars = next(iter(in_soln_inf.values()))[1][1]

    # Define the locations of the points where we are interpolating and later
    # spanwise averaging.
    coord_to_idx = {'x':0, 'y':1, 'z':2}
    otherdir_direction = [d for d in coord_to_idx.keys() if d is not args.streamwise_direction and d is not args.spanwise_direction][0]
    mmin = bounds[coord_to_idx[args.streamwise_direction], 0]
    mmax = bounds[coord_to_idx[args.streamwise_direction], 1]
    str_stations = np.linspace(mmin, mmax, num=args.n_streamwise_stations, endpoint=True)

    mmin = bounds[coord_to_idx[args.spanwise_direction], 0]
    mmax = bounds[coord_to_idx[args.spanwise_direction], 1]
    spn_stations = np.linspace(mmin, mmax, num=args.n_spanwise_stations, endpoint=False)

    mmin = bounds[coord_to_idx[otherdir_direction], 0]
    mmax = bounds[coord_to_idx[otherdir_direction], 1]
    oth_stations = np.linspace(mmin, mmax, num=args.n_otherdir_stations, endpoint=True)

    #Put them together. i am not smart enough to do this properly
    recv_loc = np.empty((ndims,
                         args.n_streamwise_stations,
                         args.n_spanwise_stations,
                         args.n_otherdir_stations))
    for i in range(args.n_streamwise_stations):
        icoord = str_stations[i]
        for j in range(args.n_spanwise_stations):
            jcoord = spn_stations[j]
            for k in range(args.n_otherdir_stations):
                kcoord = oth_stations[k]
                recv_loc[coord_to_idx[args.streamwise_direction], i, j, k] = icoord
                recv_loc[coord_to_idx[args.spanwise_direction], i, j, k] = jcoord
                recv_loc[coord_to_idx[otherdir_direction], i, j, k] = kcoord


    # Allocate the memory for the solution. Rmemebr to order it in the steamwise
    # TODO direction before writing it to file
    out_soln = np.empty((nvars, *recv_loc.shape[1:])).reshape((nvars, -1))

    # Load the elements of the input mesh: a list of lists. The outer element
    # of the list is for the partition, the inner is for the elements type.
    # Pass the in_solution only if a proper interpolation is needed rather than
    # a minimum distance search.
    # in_eles_p, in_etypes_p = get_eles(in_mesh, in_solution, in_cfg)
    in_eles_p, in_etypes_p = get_eles(in_mesh, None, in_cfg)

    #TODO take into account the possibility of not having scipy installed.

    # Build the trees of the source mesh.
    trees_partition = []
    for idxp_in, eles_in in enumerate(in_eles_p):
        print('Creating trees of input parition {}...'.format(idxp_in))
        # Get all the solution point locations for the elements
        eupts = [e.ploc_at_np('upts').swapaxes(1, 2) for e in eles_in]

        # Flatten the physical location arrays
        feupts = [e.reshape(-1, e.shape[-1]) for e in eupts]

        # For each element type construct a KD-tree of the upt locations
        trees = [cKDTree(f) for f in feupts]

        trees_partition.append((trees, eupts))

    # Get the donors for the receiver points
    pts = recv_loc.reshape((ndims, -1)).swapaxes(0,1)
    donors = []
    donors_ptrn = [] #partition number of each donor
    # Loop over the trees of the input (donor) mesh and look for donors
    for idxp_in,((trees, eupts_tree),etypes) in enumerate(zip(trees_partition, in_etypes_p)):

        # Locate the closest solution points
        closest = _closest_upts(etypes, eupts_tree, pts, trees=trees)

        if not donors and not donors_ptrn:
            # Initialize the lists.
            donors = list(closest)
            donors_ptrn = [idxp_in for d in range(len(donors))]
        else:
            # loop over the points in this partition of the receiving mesh
            for idx, (cp, dn) in enumerate(zip(closest, donors)):
                # Update donors if the distance is smaller than before
                if cp[0] < dn[0]:
                    donors[idx] = cp
                    donors_ptrn[idx] = idxp_in

    # Now that we know the donors, copy the solution
    for idx,(dn,ptrn) in enumerate(zip(donors,donors_ptrn)):
        dn_ui, dn_ei = dn[-1]
        in_sol = in_solution['{}_{}_p{}'.format(prefix, dn[-2], ptrn)]

        # # in case a proper interpolation is needed
        # dn_etype = in_etypes_p[ptrn].index(dn[-2])
        # in_sol = in_eles_p[ptrn][dn_etype]._scal_upts

        out_soln[:, idx] = in_sol[dn_ui, :, dn_ei]

    # write the new solution to file.
    out_soln = out_soln.reshape((nvars, *recv_loc.shape[1:]))
    # sol has now the shape of (nvars, nstreamwise_st, nspanwise_st, notherdir_stations)

    # Put it together with mesh
    soln = np.concatenate((recv_loc, out_soln), axis=0)

    # spanwise average
    # sol has now the shape of (nvars, nstreamwise_st, nspanwise_st, notherdir_stations)
    soln = np.mean(soln, axis=2)
    # sol has now the shape of (nvars, nstreamwise_st, notherdir_stations)
    soln = soln.reshape((ndims+nvars, -1))

    # Convert to named array.
    tps = np.float
    names = ['x', 'y', 'z'] + varnames
    soln_dtyptes = [(name, tps) for name in names]
    soln_named = np.empty(soln.shape[-1], dtype=soln_dtyptes)
    for idx, name in enumerate(names):
        soln_named[name] = soln[idx]

    #order in streamwise increasing and write to file.
    soln_named = np.sort(soln_named, order=[args.streamwise_direction, otherdir_direction])

    np.savetxt(args.outsolution, soln_named, delimiter=',',
               header=','.join(names), fmt='% .9e')

def process_import(args):
    # Get a suitable mesh reader instance
    if args.type:
        reader = get_reader_by_name(args.type, args.inmesh)
    else:
        extn = os.path.splitext(args.inmesh.name)[1]
        reader = get_reader_by_extn(extn, args.inmesh)

    # Get the mesh in the PyFR format
    mesh = reader.to_pyfrm()

    # Save to disk
    with h5py.File(args.outmesh, 'w') as f:
        for k, v in mesh.items():
            f[k] = v


def process_partition(args):
    # Ensure outd is a directory
    if not os.path.isdir(args.outd):
        raise ValueError('Invalid output directory')

    # Partition weights
    if ':' in args.np:
        pwts = [int(w) for w in args.np.split(':')]
    else:
        pwts = [1]*int(args.np)

    # Partitioner-specific options
    opts = dict(s.split(':', 1) for s in args.popts)

    # Create the partitioner
    if args.partitioner:
        part = get_partitioner(args.partitioner, pwts, order=args.order,
                               opts=opts)
    else:
        for name in sorted(cls.name for cls in subclasses(BasePartitioner)):
            try:
                part = get_partitioner(name, pwts, order=args.order)
                break
            except OSError:
                pass
        else:
            raise RuntimeError('No partitioners available')

    # Partition the mesh
    mesh, part_soln_fn = part.partition(NativeReader(args.mesh))

    # Prepare the solutions
    solnit = (part_soln_fn(NativeReader(s)) for s in args.solns)

    # Output paths/files
    paths = it.chain([args.mesh], args.solns)
    files = it.chain([mesh], solnit)

    # Iterate over the output mesh/solutions
    for path, data in zip(paths, files):
        # Compute the output path
        path = os.path.join(args.outd, os.path.basename(path.rstrip('/')))

        # Save to disk
        with h5py.File(path, 'w') as f:
            for k, v in data.items():
                f[k] = v


def process_export(args):
    # Get writer instance by specified type or outf extension
    if args.type:
        writer = get_writer_by_name(args.type, args)
    else:
        extn = os.path.splitext(args.outf)[1]
        writer = get_writer_by_extn(extn, args)

    # Write the output file
    writer.write_out()


def _process_common(args, mesh, soln, cfg):
    # Prefork to allow us to exec processes after MPI is initialised
    if hasattr(os, 'fork'):
        from pytools.prefork import enable_prefork

        enable_prefork()

    # Import but do not initialise MPI
    from mpi4py import MPI

    # Manually initialise MPI
    MPI.Init()

    # Ensure MPI is suitably cleaned up
    register_finalize_handler()

    # Create a backend
    backend = get_backend(args.backend, cfg)

    # Get the mapping from physical ranks to MPI ranks
    rallocs = get_rank_allocation(mesh, cfg)

    # Construct the solver
    solver = get_solver(backend, rallocs, mesh, soln, cfg)

    # If we are running interactively then create a progress bar
    if args.progress and MPI.COMM_WORLD.rank == 0:
        pb = ProgressBar(solver.tstart, solver.tcurr, solver.tend)

        # Register a callback to update the bar after each step
        callb = lambda intg: pb.advance_to(intg.tcurr)
        solver.completed_step_handlers.append(callb)

    # Execute!
    solver.run()

    # Finalise MPI
    MPI.Finalize()


def process_run(args):
    _process_common(
        args, NativeReader(args.mesh), None, Inifile.load(args.cfg)
    )


def process_restart(args):
    mesh = NativeReader(args.mesh)
    soln = NativeReader(args.soln)

    # Ensure the solution is from the mesh we are using
    if soln['mesh_uuid'] != mesh['mesh_uuid']:
        raise RuntimeError('Invalid solution for mesh.')

    # Process the config file
    if args.cfg:
        cfg = Inifile.load(args.cfg)
    else:
        cfg = Inifile(soln['config'])

    _process_common(args, mesh, soln, cfg)


if __name__ == '__main__':
    main()
