from setuptools import setup, find_packages

setup(
    name="BMSSP",
    version="0.1",
    packages=find_packages(),
    install_requires=["pandas~=2.3.3", "numpy~=2.2.6", "msgpack-python~=0.5.6"],
    author="@bzantium, @carlosbravo1408",
)