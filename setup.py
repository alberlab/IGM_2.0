#!/usr/bin/env python
from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy
#from setuptools import setup, find_packages
import re, os
def find_packages(path='.'):
    ret = []
    for root, dirs, files in os.walk(path):
        if '__init__.py' in files:
            ret.append(re.sub('^[^A-z0-9_]+', '', root.replace('/', '.')))
    return ret

install_requires = [
    'numpy>=1.9', 
    'scipy>=0.16', 
    'h5py>=2.5', 
    'alabtools>=0.0.1',
    'tqdm', 'tornado',
    'cloudpickle',
    'cython'
]

tests_require = [
    'mock'
]


extras_require = {
    'docs': [
        'Sphinx>=1.1', 
    ]
}

extensions = [
    Extension("igm.cython_compiled.sprite",
              ["igm/cython_compiled/sprite.pyx", "igm/cython_compiled/cpp_sprite_assignment.cpp"]),
]

extensions = cythonize(extensions)

setup(
        name = 'igm', 
        version = '1.0.0', 
        author = 'Guido Polles, Nan Hua, Lorenzo Boninsegna', 
        author_email = 'polles@usc.edu nhua@usc.edu bonimba@g.ucla.edu', 
        url = 'https://github.com/alberlab/igm', 
        description = 'Integrated Genome Modeling',
        
        packages=find_packages('./igm/'),
        package_data={'' : ['core/defaults/*', 'ui/static/css/*', 'ui/static/js/*',
                            'ui/static/js/genome_visualizer/*', 'ui/templates/*']},
        include_package_data=True,
        install_requires=install_requires,
        tests_require=tests_require,
        extras_require=extras_require,
        
        ext_modules=extensions,
        include_dirs=[numpy.get_include()],
        scripts=['bin/igm-run', 'bin/igm-server', 'bin/igm-register-dir', 'bin/igm-info-dir',
                 'bin/igm-unregister-dir', 'bin/igm-stop', 'bin/igm-report'],
)
