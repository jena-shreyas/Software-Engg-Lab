import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="package_20CS30049",
    version="0.0.1",
    author="Shreyas Jena",
    author_email="jenashreyas@gmail.com",
    description="A package that provides image segmentation and detection functionalities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jena-shreyas/Software_Engg_Lab/tree/main/Assn4/Python_DS_Assignment",
    
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"":"src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)
