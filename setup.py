from setuptools import setup
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='pyseq2500',
    version='0.61',
    description='Control an Illumina HiSeq 2500 System',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/nygctech/PySeq2500',
    author='Kunal Pandit',
    author_email ='kpandit@nygenome.org',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Operating System :: Microsoft :: Windows',
        ],
    keywords='sequencing, HiSeq, automation, biology',
    packages=['pyseq'],
    python_requires='>=3.5',
    install_requires=['pyserial>=3', #add version numbers
                      'numpy',
                      'scipy',
                      'imageio',
                      'tabulate',
                      'dask', 'xarray', 'zarr', 'qtpy','napari',
                      'scikit-image >= 0.17'],
    package_data={
        'pyseq': ['recipes/*', 'focus_data/*',
                  'focus_data/full/*', 'focus_data/partial/*',
                  'resources/*']
        },

    #package_data={  # Optional
    #    'sample': ['package_data.dat'], ## add data files inside of package
    #},
    entry_points={
        'console_scripts': [
            'pyseq = pyseq.main:main',
            ],
        },
    project_urls={
        'Bug Reports': 'https://github.com/nygctech/PySeq2500/issues',
        'Support':'https://www.hackteria.org/wiki/HiSeq2000_-_Next_Level_Hacking#Control_Software',
        'Source': 'https://github.com/nygctech/PySeq2500'
        },
)
