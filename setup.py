from setuptools import setup, Extension

"""
Note
----

This document should contain information only regarding the Cython 
extension. The information regarding the project in general (name, 
version, author, decription etc.) is stored in the `setup.cfg` file, 
as recommended from the official setuptools documentation.
"""

ext = Extension(
    name='signaturemean.cutils',
    sources=['src/signaturemean/cutils.pyx'], # if not working, try `sources=['src/signaturemean/cutils.pyx']`
)

setup(
#    name='signaturemean',

#    setup_requires=[
#        'setuptools>=18.0',  # automatically handles Cython extensions
#        'cython>=0.28.4',
#    ], # Using setup_requires is discouraged in favor of PEP-518.
    ext_modules=[ext],
#    packages=[
#        'signaturemean',
#        'signaturemean.barycenters',
#        'signaturemean.clustering'
#    ]
)
