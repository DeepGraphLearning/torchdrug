import setuptools


if __name__ == "__main__":
    setuptools.setup(
        name="torchdrug",
        description="A powerful and flexible machine learning platform for drug discovery",
        version="0.1.0",
        license="Apache-2.0",
        packages=setuptools.find_packages(),
        package_data={
            "torchdrug": [
                "layers/functional/extension/*.h",
                "layers/functional/extension/*.cuh",
                "layers/functional/extension/*.cpp",
                "layers/functional/extension/*.cu",
                "utils/extension/*.cpp",
                "utils/template/*.html"
            ]},
        test_suite="nose.collector",
        install_requires=
            [
                "torch>=1.4.0",
                "torch-scatter>=1.4.0",
                "decorator",
                "numpy>=1.11",
                "matplotlib",
                "tqdm",
                "networkx",
                "jinja2",
            ],
        python_requires=">=3.5,<3.9",
    )