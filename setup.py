from setuptools import setup, find_packages

setup(name='qwixx_gym',
      use_scm_version=True,
      setup_requires=['setuptools_scm'],
      install_requires=['gym'],
      packages=find_packages(),
      include_package_data=True,
      )
