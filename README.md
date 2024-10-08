# Data driven supervisory control with spectral learning
This repository includes the implementation of supervisors for discrete event systems againt several types of attacks and traffic light simulation example.
## Dependencies
 - ```Python``` (>=3.5)
 - ```OpenFst```(>=1.6.1) with ```pywrapfst``` module
 
The examples in this repository also require the following libraries for visualization.
 - ```Jupyter``` (>=1.0)
 - ```Graphviz```(>=2.36)
 
#### Ubuntu 18.04
You can install the ```OpenFst``` library from the Ubuntu repository:
```
sudo apt install libfst-dev
```
and you can install the Python wrapper for ```OpenFst``` using ```pip```. The version ```openfst==1.6.1``` is currently the only version compatible with ```libfst-dev```.
```
pip3 install openfst==1.6.1
```

#### Ubuntu and macOS
You need to compile the ```OpenFst``` library first with ```--enable-far``` option.
```
curl -O http://www.openfst.org/twiki/pub/FST/FstDownload/openfst-1.6.9.tar.gz
tar -xzf openfst-1.6.9.tar.gz
cd openfst-1.6.9
./configure --enable-far
make -j4
sudo make install
```
After that you can simply install the Python wrapper using ```pip```.
```
pip3 install openfst
```

#### Windows 10
You can download and install the standalone Python wrappers for 64-bit Windows from the following links:

|  Python Version  | File |
| ---------------- |:-------------:|
| ```Python 3.5``` | [pywrapfst-1.6.9-cp35-cp35m-win_amd64.whl](https://users.cs.duke.edu/~abozkurt/misc/pywrapfst-win/pywrapfst-1.6.9-cp35-cp35m-win_amd64.whl) |
| ```Python 3.6``` | [pywrapfst-1.6.9-cp36-cp36m-win_amd64.whl](https://users.cs.duke.edu/~abozkurt/misc/pywrapfst-win/pywrapfst-1.6.9-cp36-cp36m-win_amd64.whl) |
| ```Python 3.7``` | [pywrapfst-1.6.9-cp37-cp37m-win_amd64.whl](https://users.cs.duke.edu/~abozkurt/misc/pywrapfst-win/pywrapfst-1.6.9-cp37-cp37m-win_amd64.whl) |

After that you can simply install the Python wrapper using ```pip```.
```
pip3 install pywrapfst-1.6.9-cp36-cp36m-win_amd64.whl
```

#### Graphviz and Jupyter (Optional)
Our examples in this repository require ```Graphviz``` and ```Jupyter``` for visualization. You can install ```Jupyter``` using ```pip```. Example:
```
pip3 install jupyter
```
You can find detailed installation instructions for ```Graphviz``` [here](https://www.graphviz.org/download/).

## Installation
To install the current release:
```
git clone https://github.com/alperkamil/arsc.git
cd arsc
pip3 install .
```
## Basic Usage
You can access examples in the notebook titled "TL_example"


