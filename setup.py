import os
import setuptools

# Change directory to allow installation from anywhere
script_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_folder)

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

# Installs
setuptools.setup(
    name="sledge",
    version="0.1",
    author="University of Tuebingen",
    author_email="daniel.dauner@uni-tuebingen.de",
    description="Simulation Environments for Vehicle Motion Planning with Generative Models.",
    url="https://github.com/autonomousvision/sledge",
    python_requires=">=3.9",
    packages=["sledge"],
    package_dir={"": "."},
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
        "License :: Free for non-commercial use",
    ],
    license="apache-2.0",
    install_requires=requirements,
)
