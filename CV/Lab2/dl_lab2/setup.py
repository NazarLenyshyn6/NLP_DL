from setuptools import setup, find_packages

setup(
    name="dl_lab2_setup",  
    version="0.1.0", 
    author_email="nazarleny01@gmail.com",
    description="Solution for dl lab  2", 
    url="https://github.com/NazarLenyshyn6/NLP_DL/tree/main",  
    packages=find_packages(), 
    install_requires=[
        "numpy", 
        "pandas",
    ],
    python_requires=">=3.7",
)
