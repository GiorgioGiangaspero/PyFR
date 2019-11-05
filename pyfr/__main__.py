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

    #Map command
    ap_map = sp.add_parser('map', help='map --help')
    ap_map.add_argument('inmesh', type=str, help='input mesh file')
    ap_map.add_argument('insolution', type=str,
                                 help='input solution file')
    ap_map.add_argument('outmesh', type=str,
                                help='output PyFR mesh file')
    ap_map.add_argument('outsolution', type=str,
                                help='output solution file')
    ap_map.set_defaults(process=process_map)

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

def _closest_el(pts, tree, name):
    import numpy as np

    # Query the distance/index of the closest pts
    dmins, amins = tree.query(pts)

    # Return a structured array
    closest = np.empty(dmins.shape[0], dtype=[('d', np.float64),
                                              ('i', np.int),
                                              ('n', 'U256')])
    closest['d'] = dmins
    closest['i'] = amins
    closest['n'] = name

    return closest

def process_map(args):
    from collections import OrderedDict
    import numpy as np
    import itertools

    try:
        from scipy.spatial import cKDTree
    except ImportError:
        raise RuntimeError('Process Map requires the scipy package')


    # Read the input mesh
    in_mesh = NativeReader(args.inmesh)

    # #Get the number of elements of each type in each partition: meshinfo[etype][part]
    # in_meshinfo = in_mesh.partition_info('spt')

    # Read the output mesh
    out_mesh = NativeReader(args.outmesh)
    out_mesh_inf = out_mesh.array_info('spt')

    # Read the in solution
    in_solution = NativeReader(args.insolution)

    # Read the in config from solution file
    in_cfg = Inifile(in_solution['config'])

    # Get the prefix and the number of variables stored in the input solution
    # The data file prefix defaults to soln for backwards compatibility
    stats_in = Inifile(in_solution['stats'])
    prefix = stats_in.get('data', 'prefix', 'soln')

    in_mesh_inf = in_mesh.array_info('spt')
    in_soln_inf = in_solution.array_info(prefix)

    ndims = next(iter(in_mesh_inf.values()))[1][2]
    nvars = next(iter(in_soln_inf.values()))[1][1]

    # Load the elements of the input mesh: a list of lists. The outer element
    # of the list is for the partition, the inner is for the elements type.
    # Pass the in_solution only if a proper interpolation is needed rather than
    # a minimum distance search.
    # in_eles_p, in_etypes_p = get_eles(in_mesh, in_solution, in_cfg)
    in_eles_p, in_etypes_p = get_eles(in_mesh, None, in_cfg)

    # Load the elements of the output mesh
    out_eles_p, out_etypes_p = get_eles(out_mesh, None, in_cfg)

    # Create the solution map for output
    solnmap = OrderedDict()
    solnmap['mesh_uuid'] = out_mesh['mesh_uuid']
    solnmap['stats'] = stats_in.tostr().encode()
    solnmap['config'] = in_cfg.tostr().encode()

    # Open the file and write what we have so far.
    msh5 = h5py.File(args.outsolution, 'w-')
    for k, v in solnmap.items():
        msh5.create_dataset(k, data=v)

    # Build the trees of the source mesh.
    trees_partition = []
    print('Creating trees of input partitions...')
    for i_p_name, (i_etype, i_shape) in in_mesh_inf.items():
        # Get the centers of each element
        i_eupts = np.mean(in_mesh[i_p_name], axis=0)

        # For each element type construct a KD-tree of the vertices
        i_tree = cKDTree(i_eupts)

        trees_partition.append((i_p_name, i_tree, i_eupts, i_etype))

    # Loop over the partitions of the output mesh.
    for o_p_name, (o_etype, _) in out_mesh_inf.items():
        print('Working on out partition {}...'.format(o_p_name))

        # Get the elements of this partition
        pn = int(o_p_name.split('_')[-1].replace('p',''))
        oe = out_eles_p[pn][out_etypes_p[pn].index(o_etype)]

        # Get the centers of each element
        o_eupts = np.mean(out_mesh[o_p_name], axis=0)

        if o_eupts.shape[0] != oe.neles:
            raise RuntimeError('Mesh and sol array info do not match')

        donors = np.empty(oe.neles, dtype=[('d', np.float64), ('i', np.int),
                                           ('n', 'U256')] )

        # Loop over the (input mesh) trees and look for donors
        for iii,(i_p_name, i_tree, i_eupts, i_etype) in enumerate(trees_partition):
            # print('Looking for donors in partition {}'.format(i_p_name))
            closest = _closest_el(o_eupts, i_tree, i_p_name)

            if iii == 0:
                    # Initialize the lists.
                    donors[:] = closest[:]
            else:
                # print('Updating the donor info in partition {}'.format(i_p_name))
                # Update the donors if the distance is smaller then before
                ii = np.argwhere(closest['d'] < donors['d'])
                donors[ii] = closest[ii]

        # Check that we got zero distance
        if not np.allclose(donors['d'], 0.0):
            raise RuntimeError('The closest point distance is not zero everywhere.')

        # Now that we know the donor elements for this partition, copy the solution.
        # print('Creating the solution for the output partition {}'.format(o_p_name))
        out_soln = np.empty((oe.nupts, nvars, oe.neles))

        # Group by donor partition to speed things up.
        isrt = np.argsort(donors['n'])
        offset = 0
        for dp,group in itertools.groupby(donors['n'][isrt]):
            in_sol = in_solution[dp.replace('spt', prefix)]

            nd = len(list(group))
            ii = isrt[offset:offset + nd]

            out_soln[...,ii] = in_sol[..., donors['i'][ii]]

            offset += nd

        # save the partition in the solution dictionary.
        msh5.create_dataset(o_p_name.replace('spt', prefix),
                                data=out_soln)

    msh5.close()

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
