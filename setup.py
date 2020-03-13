from setuptools import setup, find_packages
from os import path

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='PySeq',
    version='0.1',
    url='https://github.com/nygctech/PySeq2500V2',
    author='Kunal Pandit',
    classifiers=[
        'Development Status :: 3 - Alpha',
        #'License ::' ADD LICENSE
        'Programming Language :: Python :: 3'
        ],
    keywords='sequencing, HiSeq, automation, biology',
    package_dir={'':: 'src'},
    python_requires='>=3.5',
    install_requires=['pyserial'],
    #package_data={ MAY WA?NT TO INCLUDE PSFS
    #entry_points ???
    #project_urls={ ADD HACKTERIA},
)
