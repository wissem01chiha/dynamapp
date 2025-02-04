Installation 
======  

Windows
----

> **Note:** If Anaconda is installed on your system, skip the first step. Conda is the only Python package manager supported at the moment.


1. Download and install [Miniconda](https://docs.anaconda.com/miniconda/miniconda-install/). For Windows installation, refer to the [Miniconda Windows Installation Guide](https://docs.anaconda.com/miniconda/miniconda-install/).

2. Install the package using the following command:
    ```shell
    pip install -i https://pypi.anaconda.org/chiha/simple pydynamapp
    ```

#### Using Git:

1. Clone the repository:  
    ```shell
    git clone https://github.com/wissem01chiha/dynamic-identification
    ```
2. Install the following enviroment dependencies manually:  
    - `numpy==1.26.4`
    - `pinocchio==2.7.1`
    - `scipy==1.14.0`
    - `nlopt==2.7.1`
    - `pyyaml==6.0.2`
    - `pandas==2.2.2`
    - `seaborn==0.13.2`
    - `matplotlib==3.8.4`   

    -- using conda:
    ```shell
    conda config --add channels conda-forge
    conda install pinocchio=2.7.1
    conda install nlopt=2.7.1
    ...
    ```
    -- or for Windows platforms, run the command to install all requirements:
    ```shell
     conda env create -f environment.yml 
    ```
    by defualt it create a virtual enviroment named **dynamapp** and place the python packages in it.
   
> **⚠️ Warning:** The Pinocchio release 2.7.1 is the officially supported version. Newer releases (3.x and above) may cause errors. If you encounter any issues, please open an [issue](https://github.com/wissem01chiha/dynamic-identification/issues).
    

3. Run all tests to check the installation:

    ```shell
    python3 run_tests.py
    ```

for any bugs encountered during installation, please make sure to open an issue at: [Issues](https://github.com/wissem01chiha/dynamic-identification/issues)


Unix
----

 We strongly recommend using conda to create a virtual environment with: 
 
```shell
conda env create -f environment.yml
conda activate dynamapp
```
an then follow the same instruction in [using git](#installation/usinggit) to insatll code dependancies in the virtual enviroment.
run test to check installation: 
.. on Linux platforms:
```shell 
  chmod u+x run_tests.sh
  ./run_tests.sh
``` 
for any bugs encountered during installation, please make sure to open an issue at: [Issues](https://github.com/wissem01chiha/dynamic-identification/issues)

> **⚠️ Warning:** The package has not been tested or built for Unix-like systems (macOS, Linux).