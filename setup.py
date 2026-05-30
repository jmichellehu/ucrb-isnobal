from setuptools import setup, find_packages

setup(
    name='isnobal_eval',
    version='0.1.0',
    packages=find_packages(include=['isnobal_eval', 'isnobal_eval.*']),
    python_requires='>=3.9',
    install_requires=[
        'xarray',
        'zarr',
        'numpy',
        'pandas',
        'geopandas',
        'matplotlib',
        'pyyaml',
        'pydantic',
    ],
    extras_require={
        'dashboard': ['panel', 'hvplot', 'holoviews'],
    },
)
