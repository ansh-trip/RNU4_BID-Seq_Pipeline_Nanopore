from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    requirements = [
        l.strip() for l in f
        if l.strip() and not l.startswith("#") and not l.strip().startswith("# ")
    ]

setup(
    name="rnu4-bidseq-pipeline",
    version="1.0.0",
    author="",
    description=(
        "Nanopore-adapted BID-seq pipeline for pseudouridine mapping and "
        "RENU syndrome variant detection in RNU4-2"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    package_data={
        "bidseq": ["../templates/*.j2", "../config/*.yaml"],
    },
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "bidseq=bidseq.cli:main",
        ],
    },
    python_requires=">=3.10",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: POSIX :: Linux",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
