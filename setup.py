from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ["tensorflow-gpu", "gym", "numpy>=1.13"]


setup( name='abp',
       version='0.1',
       description='Adaptation Based Programming Library in python',
       author='Magesh Kumar',
       author_email='muralim@oregonstate.edu',
       include_package_data=True,
       packages=find_packages(),
       install_requires=REQUIRED_PACKAGES
     )
