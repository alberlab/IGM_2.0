from __future__ import division, print_function

from alabtools.utils import Genome, Index, make_diploid, make_multiploid
from alabtools.analysis import HssFile, COORD_DTYPE
from alabtools import Contactmatrix
import os.path
import json
from six import string_types, raise_from
import numpy as np
import os
from shutil import copyfile

#===prepare genome and index instances
def PrepareGenomeIndex(cfg):

    """ Prepare genome and index instances, use Alabtools utilities

	cfg: configuration file object, as from igm.config() 
    """
    
    gcfg = cfg['genome']
    if 'usechr' not in gcfg:
        gcfg['usechr'] = ['#', 'X', 'Y']

    genome = Genome(gcfg['assembly'], usechr=gcfg['usechr'])

    if isinstance(gcfg['segmentation'], string_types):
        if os.path.isfile(gcfg['segmentation']):
            index = Index(gcfg['segmentation'], genome=genome)
        else:
            try:
                gcfg['segmentation'] = int(gcfg['segmentation'])
            except ValueError:
                raise_from(ValueError('Invalid segmentation value (either the file is not found or it is not an integer)'), None)

    if isinstance(gcfg['segmentation'], int):
        index = genome.bininfo(gcfg['segmentation'])

    if (not isinstance(gcfg['ploidy'], string_types)) and (not isinstance(gcfg['ploidy'], dict)):
        raise ValueError('Invalid ploidy value')

    if isinstance(gcfg['ploidy'], string_types):
        if gcfg['ploidy'] == 'diploid':
            index = make_diploid(index)
        elif gcfg['ploidy'] == 'haploid':
            pass
        elif gcfg['ploidy'] == 'male':
            gcfg['ploidy'] = {
                '#': 2,
                'X': 1,
                'Y': 1
            }
        else:
            gcfg['ploidy'] = json.parse(gcfg['ploidy'])

    if isinstance(gcfg['ploidy'], dict):
        chrom_ids = []
        chrom_mult = []
        for c in sorted(gcfg['ploidy'].keys()):
            if c == '#':
                autosomes = [ i for i, x in enumerate(genome.chroms)
                              if x[-1].isdigit() ]
                chrom_ids += autosomes
                chrom_mult += [ gcfg['ploidy'][c] ] * len(autosomes)
            else:
                if isinstance(c, string_types):
                    cn = genome.chroms.tolist().index('chr%s' % c)
                elif isinstance(c, int):
                    cn = c
                else:
                    raise ValueError('Invalid chromosome ID in ploidy: %s' % repr(cn))
                chrom_ids += [ cn ]
                chrom_mult += [ gcfg['ploidy'][c] ]
        index = make_multiploid(index, chrom_ids, chrom_mult)

    return genome, index

def prepareHss(fname, nbead, nstruct, genome, index, radii, nucleus_shape='sphere', nucleus_parameters=5000.0, nucleus_volume=0, coord_chunks=None):
    with HssFile(fname, 'w') as hss:
        #put everything into hssFile
        hss.set_nbead(nbead)
        hss.set_nstruct(nstruct)
        hss.set_genome(genome)
        hss.set_index(index)
        hss.set_radii(radii)
        if coord_chunks is None:
            hss.set_coordinates(np.zeros((nbead,nstruct,3)))
        else:
            hss.create_dataset('coordinates', shape=(nbead, nstruct, 3), dtype=COORD_DTYPE, chunks=coord_chunks)
        env = hss.create_group('envelope')
        env.create_dataset('shape', data=nucleus_shape)
        env.create_dataset('volume', data=nucleus_volume)
        env.create_dataset('params', data=nucleus_parameters)

def Preprocess(cfg):

    #Generate genome, index objects
    genome, index = PrepareGenomeIndex(cfg)

    # number of structures in population, number of beads
    nstruct = cfg['model']['population_size']
    nbead = len(index)

    # read in volume occupancy as from CFG file
    occupancy = cfg['model']['occupancy']
    _43pi = 4./3*np.pi

    # compute volume of the nucleus
    nucleus_shape = cfg.get('model/restraints/envelope/nucleus_shape')
    if nucleus_shape == 'sphere':
        nucleus_radius     = cfg.get('model/restraints/envelope/nucleus_radius')
        nucleus_volume     = _43pi * (nucleus_radius**3)
        nucleus_parameters = nucleus_radius
    elif nucleus_shape == 'ellipsoid':
        sx = cfg.get('model/restraints/envelope/nucleus_semiaxes')
        nucleus_volume     = _43pi * sx[0] * sx[1] * sx[2]
        nucleus_parameters = sx
    elif nucleus_shape == 'exp_map':      # account for a random nucleus, initialize on a sphere
        nucleus_radius     = 5100         # this is the effective radius for the volume map/w nucleolus
        nucleus_volume     = _43pi * (nucleus_radius**3)
        nucleus_parameters = nucleus_radius
    else:
        raise NotImplementedError(
            "Cannot compute volume for shape %s" % cfg.get('model/restraints/envelope/nucleus_shape')
        )

    # compute volume per basepair
    rho = occupancy * nucleus_volume / (sum(index.end - index.start))
    bp_sizes = index.end - index.start
    sphere_volumes = [rho * s for s in bp_sizes]
    radii = ( np.array(sphere_volumes) / _43pi )**(1./3)

    # prepare Hss
    if not os.path.isfile(cfg['optimization']['structure_output']):
        prepareHss(cfg['optimization']['structure_output'], nbead, nstruct, genome, index, radii, nucleus_shape, nucleus_parameters, nucleus_volume)

    # now create a temporary struct-major file for runtime use
    if not os.path.isfile(cfg['optimization']['structure_output'] + '.T'):

        PACK_SIZE = 1e6
        pack_struct = max(1, int( PACK_SIZE / nbead / 3 ) )
        pack_struct = min(pack_struct, nstruct)

        prepareHss(cfg['optimization']['structure_output'] + '.T' , nbead,
                   nstruct, genome, index, radii, nucleus_shape,
                   nucleus_parameters, nucleus_volume,
                   coord_chunks=(nbead, pack_struct, 3))

    # prepare tmp file dir
    if not os.path.exists(cfg['parameters']['tmp_dir']):
        os.makedirs(cfg['parameters']['tmp_dir'])

    # if we have a Hi-C probability matrix, use it to determine the consecutive
    # beads distances
    pbs = cfg.get('model/restraints/polymer/polymer_bonds_style')
    if pbs == 'hic':
        if "Hi-C" not in cfg['restraints']:
            raise RuntimeError('Hi-C restraints specifications are missing in the cfg, but "polymer_bond_style" is set to "hic"')
        # read the HiC matrix and get the first diagonal.
        m = Contactmatrix(cfg['restraints']['Hi-C']['input_matrix']).matrix
        cps = np.zeros(len(index) - 1)
        for i in range(m.shape[0] - 1):
            f = m[i][i+1]
            for j in index.copy_index[i]:
                cps[j] = f
        cpfname = os.path.join(cfg['parameters']['tmp_dir'], 'consecutive_contacts.npy')
        np.save(cpfname, cps)
        cfg['runtime']['consecutive_contact_probabilities'] = cpfname







