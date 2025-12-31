from setuptools import find_packages, setup
from typing import List

Hypen_E_DOT='-e .'

def get_requirements(file_path: str) -> list[str]:
    """This function will return the list of requirements"""
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace('\n', '') for req in requirements]
        if Hypen_E_DOT in requirements:
            requirements.remove(Hypen_E_DOT)
        
    return requirements



setup(
name="ML_project",
version="0.0.1",
author="Kartikey",
author_email="kartikeytripathi990@gmail.com",
packages=find_packages(),
install_requires=get_requirements("requirements.txt"),
)