import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
     name='svwcbs',  
     version='0.1',
     author="Simon van Wageningen",
	 install_requires=[
        'networkx~=2.3',
        'numpy~=1.17.2',
        'scipy~=1.3.1'
    ],
     author_email="simon.van.wageningen@outlook.com",
     description="Package for CBS network complexities",
     long_description=long_description,
   long_description_content_type="text/markdown",
     url="https://github.com/javatechy/dokr",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )
