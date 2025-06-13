from setuptools import find_packages, setup

setup(
    name="odor-pair",
    version="1.0",
    author="Laura Sisson",
    description="Tools for modeling odorant pairs",
    packages=find_packages(include=["odorpair", "odorpair.*"]),
    python_requires=">=3.11",
    install_requires=[
        "torch",
        "torch-geometric",
        "rdkit",
        "ogb",
        "numpy"
    ],
    include_package_data=True,
    package_data={
        "odorpair": ["Production/*/*.json", "Production/*/*.pt"],
    },
)
