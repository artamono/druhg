import warnings

try:
    from Cython.Distutils import build_ext
    from setuptools import setup, Extension
    HAVE_CYTHON = True
except ImportError as e:
    warnings.warn(e.args[0])
    from setuptools import setup, Extension
    from setuptools.command.build_ext import build_ext
    HAVE_CYTHON = False


class CustomBuildExtCommand(build_ext):
    """build_ext command for use when numpy headers are needed."""

    def run(self):

        # Import numpy here, only when headers are needed
        import numpy

        # Add numpy headers to include_dirs
        self.include_dirs.append(numpy.get_include())

        # Call original build_ext command
        build_ext.run(self)


_hdbscan_tree = Extension('druhg._hdbscan_tree',
                          sources=['druhg/_hdbscan_tree.pyx'])
_druhg_linkage = Extension('druhg._druhg_linkage',
                             sources=['druhg/_druhg_linkage.pyx'])
_prediction_utils = Extension('druhg._prediction_utils',
                              sources=['druhg/_prediction_utils.pyx'])
_druhg_boruvka = Extension('druhg._druhg_boruvka',
                             sources=['druhg/_druhg_boruvka.pyx'])
_druhg_prims = Extension('druhg._druhg_prims',
                             sources=['druhg/_druhg_prims.pyx'])
_druhg_even_rankability = Extension('druhg._druhg_even_rankability',
                                    sources=['druhg/_druhg_even_rankability.pyx'])
dist_metrics = Extension('druhg.dist_metrics',
                         sources=['druhg/dist_metrics.pyx'])


def readme():
    with open('README.rst') as readme_file:
        return readme_file.read()

def requirements():
    # The dependencies are the same as the contents of requirements.txt
    with open('requirements.txt') as f:
        return [line.strip() for line in f if line.strip()]

configuration = {
    'name': 'druhg',
    'version': '0.8.18',
    'description': 'Clustering based on density with variable density clusters',
    'long_description': readme(),
    'classifiers': [
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved',
        'Programming Language :: C',
        'Programming Language :: Python',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: MacOS',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
    ],
    'keywords': 'cluster clustering density hierarchical',
    'url': 'http://github.com/scikit-learn-contrib/druhg',
    'maintainer': 'Leland McInnes',
    'maintainer_email': 'leland.mcinnes@gmail.com',
    'license': 'BSD',
    'packages': ['druhg', 'druhg.tests'],
    'install_requires': requirements(),
    'ext_modules': [_hdbscan_tree,
                    _druhg_linkage,
                    _druhg_boruvka,
                    _druhg_prims,
                    _druhg_even_rankability,
                    _prediction_utils,
                    dist_metrics],
    'cmdclass': {'build_ext': CustomBuildExtCommand},
    'test_suite': 'nose.collector',
    'tests_require': ['nose'],
    'data_files': ('druhg/dist_metrics.pxd',)
}

if not HAVE_CYTHON:
    warnings.warn('Due to incompatibilities with Python 3.7 druhg now'
                  'requires Cython to be installed in order to build it')
    raise ImportError('Cython not found! Please install cython and try again')

setup(**configuration)
