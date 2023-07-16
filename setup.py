import setuptools


with open("README.md", "r") as fin:
    long_description = fin.read()


if __name__ == "__main__":
    setuptools.setup(
        name="torchdrug",
        description="A powerful and flexible machine learning platform for drug discovery",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://torchdrug.ai/",
        author="TorchDrug Team",
        version="0.2.1",
        license="Apache-2.0",
        keywords=["deep-learning", "pytorch", "drug-discovery"],
        packages=setuptools.find_packages(),
        package_data={
            "torchdrug": [
                "layers/functional/extension/*.h",
                "layers/functional/extension/*.cuh",
                "layers/functional/extension/*.cpp",
                "layers/functional/extension/*.cu",
                "utils/extension/*.cpp",
                "utils/template/*.html",
            ]
        },
        test_suite="nose.collector",
        install_requires=[
            "torch>=1.8.0",
            "torch-scatter>=2.0.8",
            "torch-cluster>=1.5.9",
            "decorator",
            "numpy>=1.11",
            "rdkit-pypi>=2020.9",
            "matplotlib",
            "tqdm",
            "networkx",
            "ninja",
            "jinja2",
            "lmdb",
            "fair-esm",
        ],
        python_requires=">=3.7,<3.11",
        classifiers=[
            "Development Status :: 4 - Beta",
            'Intended Audience :: Developers',
            'Intended Audience :: Science/Research',
            "License :: OSI Approved :: Apache Software License",
            'Topic :: Scientific/Engineering :: Mathematics',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'Topic :: Software Development :: Libraries',
            "Programming Language :: Python :: 3",
        ],
    )