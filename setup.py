from setuptools import setup, find_packages

setup(
    name="Excited_States_NN",                # Replace with your package name
    version="0.1.0",                 # Initial version
    author="Christian C. Schmidt",              # Your name
    author_email="christian.schmidt@kit.edu",  # Your email
    description="Excited States NN for Excitation Energies and Forces",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Br4iNd4m4gE/excited_states_tcb",  # Project URL
    packages=find_packages(),        # Automatically finds all packages in `mypackage/`
    install_requires=[
        # Extracted from `pip:` section
    ],
    classifiers=[                    # Metadata for PyPI
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires='>=3.8.8',         # Python version requirement
)
