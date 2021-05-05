
#!/usr/bin/env python

# Copyright (C) 2016 University of Southern California and
#                        Guido Polles
#
# Authors: Guido Polles
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
The **lammps** module provides function to interface
with LAMMPS in order to perform the modeling
of the chromosome structures.

To perform a full modeling step, you may
want to check out the higher level wrappers
in the wrappers module.
"""


from __future__ import print_function, division
import os
import os.path
import errno
import math
from itertools import groupby
from copy import deepcopy

from subprocess import Popen, PIPE

from .lammps_io import get_info_from_log, get_last_frame
from .lammps_model import *

try:
    UNICODE_EXISTS = bool(type(unicode))
except NameError:
    unicode = str


__author__ = "Guido Polles, Lorenzo Boninsegna"
__license__ = "GPL"
__version__ = "0.0.1"
__email__ = "polles@usc.edu, bonimba@g.ucla.edu"


INFO_KEYS = ['final-energy', 'pair-energy',
             'bond-energy', 'md-time', 'n_restr', 'n_hic_restr']
EPS_TEMP = 0.001


def create_lammps_data(model, user_args):

    """ model = lammps model already """

    n_atom_types = len(model.atom_types)
    n_bonds      = len(model.bonds)
    n_bondtypes  = len(model.bond_types)
    n_atoms      = len(model.atoms)

    # define simulation box size
    boxdim = 0
    for atom in model.atoms:
        boxdim = max(boxdim, np.max(np.abs(atom.xyz)))
    boxdim *= 1.05

    with open(user_args['data'], 'w') as f:

        print('LAMMPS input\n', file=f)

        print(n_atoms, 'atoms\n', file=f)
        print(n_atom_types, 'atom types\n', file=f)
        print(n_bondtypes, 'bond types\n', file=f)
        print(n_bonds, 'bonds\n', file=f)

        # write simulation box size(s)
        print('-{} {} xlo xhi\n'.format(boxdim, boxdim),
              '-{} {} ylo yhi\n'.format(boxdim, boxdim),
              '-{} {} zlo zhi'.format(boxdim, boxdim), file=f)

        # --- atom coordinates -----
        print('\nAtoms\n', file=f)
        # index, molecule, atom type, x y z.
        for atom in model.atoms:
            print(atom, file=f)

        # --- information about the bonds ----
        # Harmonic Upper Bond Coefficients are one for each bond type
        # and coded as:
        #   Bond_type_id kspring activation_distance
        # Each bond is coded as:
        #   Bond_id Bond_type_id ibead jbead

        if n_bonds > 0:
            print('\nBond Coeffs\n', file=f)
            for bt in model.bond_types:
                print(bt, file=f)

            print('\nBonds\n', file=f)
            for bond in model.bonds:
                print(bond, file=f)

        # Excluded volume coefficients: one set of coefficients for EACH pair-flavors (i.e., pair of particle types)
        atom_types = list(model.atom_types.values())

        print('\nPairIJ Coeffs\n', file=f)
        for i in range(len(atom_types)):
            a1 = atom_types[i]
            for j in range(i, len(atom_types)):
                a2 = atom_types[j]
                id1 = min(a1.id + 1, a2.id + 1)
                id2 = max(a1.id + 1, a2.id + 1)

                # if either one particle is not a BEAD type, no excluded volume is applied
                if (a1.atom_category == AtomType.BEAD and
                        a2.atom_category == AtomType.BEAD):
                    ri = a1.radius
                    rj = a2.radius
                    dc = (ri + rj)
                    A = (dc / math.pi)**2
                    # sigma = dc / 1.1224 #(2**(1.0/6.0))
                    #print(i+1, user_args['evfactor'], sigma, dc, file=f)

                    print(id1, id2, A * model.evfactor, dc, file=f)
                else:
                    print(id1, id2, 0.0, 0.0, file=f)

        # User data, add bead radius
        print('\nUser\n', file=f)
        for i, atom in enumerate(model.atoms):
            if hasattr(atom.atom_type, 'radius'):
                r = atom.atom_type.radius
            else:
                r = 0
            print(i + 1, atom.mol_id, r, file=f)


def create_lammps_script(model, user_args):

    """ This function prepares and writes the Lammps script which contains the 
        protocol to define the system (particles and interactions) and 
        run simulated annealing and conjugate gradient (CJ) 

	Possible Lammps fixes are to be called here
    """

    # get a seed, different for each minimization and run but deterministic
    seed = (
        (
            user_args.get('seed', np.random.randint(1, 9007991))
            * model.id
            * user_args.get('step_no', np.random.randint(1, 4325237))
        ) % 9190037
    ) + 1

    maxrad = max([at.radius for at in model.atom_types if
                  at.atom_category == AtomType.BEAD])

    with open(user_args['lmp'], 'w') as f:
        print('units                 lj', file=f)
        print('atom_style            bond', file=f)
        print('bond_style  hybrid',
              'harmonic_upper_bound',
              'harmonic_lower_bound', file=f)
        print('boundary              s s s', file=f)  # non-periodic and adapts

        # Needed to avoid calculation of 3 neighs and 4 neighs
        print('special_bonds lj/coul 1.0 1.0 1.0', file=f)

        # add radii and chromosome id
        print('fix userprop all property/atom i_chainid d_radius', file=f)

        # excluded volume
        if user_args['use_gpu']:
            pair_style = 'soft/gpu'
        else:
            pair_style = 'soft'
        print('pair_style', pair_style, 2.0 * maxrad, file=f)  # global cutoff

        print('read_data', user_args['data'], 'fix userprop NULL User', file=f)
        print('mass * 1.0', file=f)

        # groups atom types by atom_category
        sortedlist = list(
            sorted(model.atom_types, key=lambda x: x.atom_category))
        groupedlist = {k: list(v) for k, v in groupby(sortedlist,
                                                      key=lambda x: x.atom_category)}

        bead_types = [str(x) for x in groupedlist[AtomType.BEAD]]
        dummy_types = [str(x)
                       for x in groupedlist.get(AtomType.FIXED_DUMMY, [])]
        centroid_types = [str(x) for x in groupedlist.get(
            AtomType.CLUSTER_CENTROID, [])]
        print('group beads type', ' '.join(bead_types), file=f)

        if dummy_types:
            print('group dummy type', ' '.join(dummy_types), file=f)
            print('neigh_modify exclude group dummy all', file=f)
        if centroid_types:
            print('group centroid type', ' '.join(centroid_types), file=f)
            print('neigh_modify exclude group centroid all', file=f)

        print('group nonfixed type', ' '.join(centroid_types + bead_types), file=f)

        print('neighbor', maxrad, 'bin', file=f)  # skin size
        print('neigh_modify every 1 check yes', file=f)
        print('neigh_modify one', user_args['max_neigh'],
              'page', 20 * user_args['max_neigh'], file=f)

        # Freeze dummy atom
        if dummy_types:
            print('fix 1 dummy setforce 0.0 0.0 0.0', file=f)

        # Integration
        # select the integrator
        print('fix integrator nonfixed nve/limit',
              user_args['max_velocity'], file=f)

        # Valid envelopes: produce envelope groups listing the atoms subjected to the force
        valid_envelopes = [
            envelope for envelope in model.envelopes if len(envelope.particle_ids)]
        for j, envelope in enumerate(valid_envelopes):
            print(
                'group envgrp{} id {}'.format(
                    j,
                    ' '.join([str(k + 1) for k in envelope.particle_ids])
                ),
                file=f
            )

        print('timestep', user_args['timestep'], file=f)

        # Region
        # print('region mySphere sphere 0.0 0.0 0.0',
        #       user_args['nucleus_radius'] + 2 * maxrad, file=f)
        # print('fix wall beads wall/region mySphere harmonic 10.0 1.0 ',
        #       2 * maxrad, file=f)

        # print('pair_modify shift yes mix arithmetic', file=f)

        # regulate output:
        print('dump   crd_dump all custom',
              user_args['write'],
              user_args['out'],
              'id type x y z fx fy fz', file=f)

        # Write the steps of simulated annealing

        # prepare protocol
        protocol = user_args.get('custom_annealing_protocol', None)
        if protocol is None:
            # default protocol is 1 step, tstart to tstop
            protocol = {
                'num_steps': 1,
                'mdsteps': [user_args['mdsteps']],
                'tstarts': [user_args['tstart']],
                'tstops': [user_args['tstop']],
                'evfactors': [1],
                'envelope_scales': [1]
            }

        nsteps     = protocol.get('num_steps')
        mdsteps    = protocol.get('mdsteps', [user_args['mdsteps']] * nsteps)
        tstarts    = protocol.get('tstarts', [user_args['tstart']] * nsteps)
        tstops     = protocol.get('tstops', tstarts)
        evfs       = protocol.get('evfactors', [1] * nsteps)
        envelopesf = protocol.get('envelope_factors', [1] * nsteps)

        assert(len(mdsteps) == len(tstarts) == len(tstops)
               == len(evfs) == len(envelopesf) == nsteps)

        print('# factors: ', envelopesf, file=f)

        for step, (md, t0, t1, evf, envf) in enumerate(zip(mdsteps, tstarts, tstops, evfs, envelopesf)):

            print('variable evprefactor equal ', evf, file=f)
            print('fix exVolAdapt%d all adapt 0' % step,
                  'pair soft a * * v_evprefactor scale yes',
                  'reset yes', file=f)

            for j, envelope in enumerate(valid_envelopes):
                if len(envelope.particle_ids):
                    if envelope.shape == 'ellipsoid':
                        print(
                            'fix envelope{0} envgrp{0} ellipsoidalenvelope'.format(
                                j),
                            ' '.join([str(x * envf)
                                      for x in envelope.semiaxes]),   # scaling semiaxes by 'envf' (>1)  factor
                            envelope.k,
                            file=f
                        )
                        print('fix_modify envelope{} energy yes'.format(j), file=f)
                    
                    elif envelope.shape == 'exp_map':
			#print(str(envelope.volume_file).format(j), file=f)
                        print(
                            'fix envelope{0} envgrp{0} volumetricrestraint '.format(j) + str(envelope.volume_file), envf , envelope.k, file=f
                        )
                        print('fix_modify envelope{} energy yes'.format(j), file=f)


                    else:
                        raise NotImplementedError(
                            'Envelope (%s) not implemented' % envelope.shape)

            relax = protocol.get('relax', None)

            if relax is not None:
                # print('fix termostat nonfixed langevin', relax['temperature'], relax['temperature'], user_args['damp'],
                #       seed + step, file=f)
                print('velocity nonfixed create', relax[
                      'temperature'], seed + step, file=f)
                # print('fix termostat nonfixed temp/berendsen', relax['temperature'], relax['temperature'], user_args['damp'], file=f)
                print('fix termostat nonfixed temp/rescale 1 ',
                      relax['temperature'], relax['temperature'], 0.1, 1, file=f)
                print('fix integrator nonfixed nve/limit',
                      relax['max_velocity'], file=f)
                print('run', relax['mdsteps'], file=f)
                # reset original integrator
                print('fix integrator nonfixed nve/limit',
                      user_args['max_velocity'], file=f)

            # print('fix termostat nonfixed langevin', t0, t1, user_args['damp'], seed + step, file=f)
            print('velocity nonfixed create', t0, seed + step, file=f)
            # print('fix termostat nonfixed temp/berendsen', t0, t1, user_args['damp'], file=f)
            print('fix termostat nonfixed temp/rescale 1 ',
                  t0, t1, 0.1, 1, file=f)

            # Thermodynamic info style for output
            print(
                'thermo_style custom step temp epair ebond ' + ' '.join([
                    'f_envelope{}'.format(i) for i in range(len(valid_envelopes))
                ]), file=f
            )
            print('thermo_modify norm no', file=f)
            print('thermo', user_args['thermo'], file=f)

            print('run', md, file=f)

            print('unfix exVolAdapt%d' % step, file=f)

        # Run minimization using Conjugate Gradient (CG)
        print('min_style cg', file=f)
        print('minimize', user_args['etol'], user_args['ftol'],
              user_args['max_cg_iter'], user_args['max_cg_eval'], file=f)

        print('info time', file=f)


def optimize(model, cfg):
    '''
    Lammps interface for minimization.

    It first creates input and data files for lammps, then
    runs the lammps executable in a process (using subprocess.Popen).

    When the program returns, it parses the output and returns the
    new coordinates, along informations on the run and on violations.

    The files created are
    - input file: `tmp_files_dir`/`run_name`.lam
    - data file: `tmp_files_dir`/`run_name`.input
    - trajectory file: `tmp_files_dir`/`run_name`.lammpstrj

    The function tries to remove the temporary files after the run, both in
    case of failure and success (unless `keep_temporary_files` is set to
    `False`). If the interpreter is killed without being able to catch
    exceptions (for example because of a walltime limit) some files could be
    left behind.

    Parameters
    ----------
    model : `igm.Model`
        Abstracted model object for simulation optimization
    cfg : dict
        A dict of configurations.

    Returns
    -------
    new_crd : numpy ndarray
        Coordinates after minimization.
    info : dict
        Dictionary with summarized info for the run, as returned
        by `lammps.get_info_from_log`.
    violations : list
        List of violations. If the `check_violations` parameter is set
        to `False`, returns an empty list.

    Raises
    ------
    RuntimeError
        If the lammps executable return code is different from 0, it raises
        a RuntimeError with the contents of the standard error.
    '''

    tmp_files_dir = cfg['optimization']['tmp_dir']
    if not os.path.isabs(cfg['optimization']['tmp_dir']):
        tmp_files_dir = os.path.join(
            cfg['parameters']['tmp_dir'], tmp_files_dir)
    try:
        os.makedirs(tmp_files_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass

    run_name             = cfg['runtime']['run_name']
    keep_temporary_files = cfg['optimization']['keep_temporary_files']
    lammps_executable    = cfg['optimization'][
        'kernel_opts']['lammps']['lammps_executable']
    run_opts             = deepcopy(cfg['optimization']['optimizer_options'])

    data_fname           = os.path.join(tmp_files_dir, run_name + '.data')
    script_fname         = os.path.join(tmp_files_dir, run_name + '.lam')
    traj_fname           = os.path.join(tmp_files_dir, run_name + '.lammpstrj')
    log_fname            = os.path.join(tmp_files_dir, run_name + '.log')

    try:

        # prepare input
        io_opts = {'out': traj_fname, 'data': data_fname, 'lmp': script_fname}
        run_opts.update(io_opts)
        run_opts.update(cfg['optimization']['kernel_opts']['lammps'])
        # this is to set the random seed
        run_opts.update({'step_no': cfg.get('runtime/step_no', 1) + 2})

        # convert python user defined "model" object into intermediate "LammpsModel" object
        m = LammpsModel(model)
 
        # write input .lam (script) and .data files to be used in the minimization
        create_lammps_data(m, run_opts)
        create_lammps_script(m, run_opts)

        # run the lammps minimization
        with open(script_fname, 'r') as lamfile:
            proc = Popen([lammps_executable, '-log', log_fname],
                         stdin=lamfile,
                         stdout=PIPE,
                         stderr=PIPE)
            output, error = proc.communicate()

        if proc.returncode != 0:
            raise RuntimeError('LAMMPS exited with non-zero exit code: %d (modelid: %d)\nOutput:\n%s\n' % (
                               proc.returncode,
                               model.id,
                               output))

        # get results
        with open(log_fname, 'r') as lf:
            info = get_info_from_log(lf)   # this is returned as result of lammps minimization file

        with open(traj_fname, 'r') as fd:
            new_crd = get_last_frame(fd)

        # updates the input model coordinates
        for i, p in enumerate(model.particles):
            p.pos = new_crd[m.imap[i]]

        if not keep_temporary_files:
            if os.path.isfile(data_fname):
                os.remove(data_fname)
            if os.path.isfile(script_fname):
                os.remove(script_fname)
            if os.path.isfile(traj_fname):
                os.remove(traj_fname)
            if os.path.isfile(log_fname):
                os.remove(log_fname)

    except:
        if not keep_temporary_files:
            if os.path.isfile(data_fname):
                os.remove(data_fname)
            if os.path.isfile(script_fname):
                os.remove(script_fname)
            if os.path.isfile(traj_fname):
                os.remove(traj_fname)
            if os.path.isfile(log_fname):
                os.remove(log_fname)
        raise

    return info
