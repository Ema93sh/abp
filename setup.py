from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ["torch", "gym", "numpy", "visdom", "tensorboardX"]


setup(name='abp',
      version='0.1',
      description='Reinforcement Learning Library in python',
      author='Magesh Kumar',
      author_email='muralim@oregonstate.edu',
      include_package_data=True,
      package_data={'': ['tasks']},
      packages=find_packages(),
      install_requires=REQUIRED_PACKAGES
      )
