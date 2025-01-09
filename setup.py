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
        "absl-py==0.12.0",
        "chardet==4.0.0",
        "colorama==0.4.4",
        "future==0.18.2",
        "gast==0.3.3",
        "google-auth==1.30.0",
        "grpcio==1.37.1",
        "idna==2.10",
        "joblib==1.0.1",
        "keras-layer-normalization==0.16.0",
        "keras-tuner==1.0.2",
        "numpy==1.18.5",
        "oauthlib==3.1.0",
        "protobuf==3.15.8",
        "pyasn1-modules==0.2.8",
        "requests==2.25.1",
        "scikit-learn==0.24.2",
        "scipy==1.4.1",
        "tabulate==0.8.9",
        "tensorboard==2.5.0",
        "tensorboard-data-server==0.6.1",
        "tensorboard-plugin-wit==1.8.0",
        "tensorflow-estimator==2.3.0",
        "tensorflow-gpu==2.3.0",
        "termcolor==1.1.0",
        "terminaltables==3.1.0",
        "threadpoolctl==2.1.0",
        "tqdm==4.60.0",
        "urllib3==1.26.4",
        "werkzeug==1.0.1",
        "wrapt==1.12.1",
        # Key Python packages from dependencies
        "aiohttp==3.8.1",
        "numpy-base==1.20.1",
        "pandas==1.4.2",
        "h5py==2.10.0",
        "matplotlib==3.3.4",
        "tensorflow==2.4.1",
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
