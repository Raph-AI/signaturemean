from setuptools import setup, Extension
# setuptools>=18.0 handles extensions
import numpy

"""
Note
----

This document should contain information only regarding the Cython
extension. The information regarding the project in general (name,
version, author, decription etc.) is stored in the `setup.cfg` file,
as recommended from the official setuptools documentation.
"""

ext1 = Extension(
    name='signaturemean.cutils',
    sources=['src/signaturemean/cutils.pyx'],
    include_dirs=[numpy.get_include()]
    )
ext2 = Extension(
    name='signaturemean.barycenters.cmean_pennec',
    sources=['src/signaturemean/barycenters/cmean_pennec.pyx'],
    include_dirs=[numpy.get_include()]
    )
ext3 = Extension(
    name='signaturemean.barycenters.mean_group',
    sources=['src/signaturemean/barycenters/mean_group_mallocerror.pyx'],
    include_dirs=[numpy.get_include()]
    )

setup(
#    name='signaturemean',

#    setup_requires=[
#        'setuptools>=18.0',  # automatically handles Cython extensions
#        'cython>=0.28.4',
#    ], # Using setup_requires is discouraged in favor of PEP-518.
    ext_modules=[ext1, ext2, ext3]
#    packages=[
#        'signaturemean',
#        'signaturemean.barycenters',
#        'signaturemean.clustering'
#    ]
)
