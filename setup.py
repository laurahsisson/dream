from setuptools import setup, find_packages

setup(
    name="odor-pair",  # pip install name (can contain dash)
    version="1.0",
    author="Laura Sisson",
    description="Tools for modeling odorant pairs",
    packages=find_packages(include=["odorpair", "odorpair.*"]),
    python_requires=">=3.11",
    install_requires=["torch", "torch-geometric", "rdkit", "ogb", "numpy"],
)
