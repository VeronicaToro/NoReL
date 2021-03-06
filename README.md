This repository is the implementation of NoReL, a no-regret algorithm for the optimal allocation of spreading factors (SF) and transmission powers (TP) in LoRa networks. NoReL is described in ["Learning How to Configure LoRa Networks with No Regret: a Distributed Approach"](https://ieeexplore.ieee.org/document/9813414) accepted at IEEE Transactions on Industrial Informatics, 2022. This implementation simulates a LoRa network and the actions taken by each node to maximize their own delivery ratio. The simulator takes into account several key factors: quasi-orthogonal spreading factors, capture effect, duty cycling and channel variation due to fading. For more details, please read the [paper](https://ieeexplore.ieee.org/document/9813414). If you use the simulator, please cite:

V. Toro-Betancur, G. Premsankar, C. -F. Liu, M. Słabicki, M. Bennis and M. D. Francesco, "Learning How to Configure LoRa Networks with No Regret: a Distributed Approach," in IEEE Transactions on Industrial Informatics, 2022, doi: 10.1109/TII.2022.3187721.

<details>

  <summary>Click here to get BibTeX entry</summary>

  ```
    @ARTICLE{Toro2022Learning,
  author={Toro-Betancur, Veronica and Premsankar, Gopika and Liu, Chen-Feng and Slabicki, Mariusz and Bennis, Mehdi and Francesco, Mario Di},
  journal={IEEE Transactions on Industrial Informatics}, 
  title={Learning How to Configure LoRa Networks with No Regret: a Distributed Approach}, 
  year={2022},
  volume={},
  number={},
  pages={1-12},
  doi={10.1109/TII.2022.3187721}}
  ```

</details>

Setting the environment
=======================

We recommend using a Python virtual environment to run the code. Please follow the instructions using the package manager of your choice: Conda or pip. 

Conda
-----

To set the environment using [Conda](https://docs.conda.io/en/latest/), simply run the following to create a Conda environment called *lora-env*.

```
conda config --append channels conda-forge
conda create --name norel-env --file requirements.txt
```

pip
---

If you prefer to set the environment using [pip](https://pypi.org/project/pip/), you can do so as follows. This was last tested with Python version 3.7.4 and 3.8.11.

First, create the pip virtual environment (here, the environment is called *.lora-env* and will be created in a folder with the same name):
```
python -m venv .norel-env
```
Activate it
```
source .norel-env/bin/activate
```
And install the dependencies
```
pip install -r requirements.txt
```
To deactivate the environment, simply run:
```
deactivate
```

Usage
=====
Clone the repository:
```
git clone https://github.com/VeronicaToro/NoReL.git
```
From the *src/* folder you can run the simulator. That is, first run
```
cd directory_where_LoRa-model_was_cloned/NoReL/src/
```
Then, run the simulator:
```
python main.py [options]
```
You can also make *main.py* executable and avoid calling Python every time. For a complete list of options run

```
python main.py --help
```
There are three options that must always be specified:
* *-s*: Spreading Factor (SF) assignment strategy. It can be either *random* or *minimum*
* *-t*: Transmission Power (TP) assignment strategy. It can be either *random*, *minimum* or *maximum*
* *-d*: Directory where the config file is located. See below for more details on the config file
* *-sd*: Simulation time given in days

```
python main.py -s random -t minimum -d path_to_config_file -sd 10
```

Config file
===========

You can run the simulator in both, uniform scenarios or user-specified networks, as follows:

Uniform scenarios
-----------------

In uniform scenarios, nodes are uniformly distributed around gateways. Users can specify the number of nodes, the number of gateways, locations of gateway(s) and distance (*radius*) within which nodes are deployed. Specify the parameters in the file *network_config_files/network_config.ini*. For instance, for a network with 2 gateways located at coordinates (500 m, 500 m) and (1000 m, 500 m), 50 nodes deployed around each gateway within a radius of 500 m from both gateways, a sample is provided below. Note that the parameters *sizeX* and *sizeY* define the whole deployment area, such that all nodes are contained in a rectangle of sizes *sizeX*  and *sizeY*. Moreover, under [network], the total *numNodes* must be equal to the sum of *numNodes* deployed around each gateway.

```
[network]
sizeX = 1500
sizeY = 1000
numNodes = 100
numGateways = 2

[gateway_0]
numNodes = 50
xLocation = 500
yLocation = 500
radius = 500

[gateway_1]
numNodes = 50
xLocation = 1000
yLocation = 500
radius = 500
```

To run the simulator with the network specifications in the file called *network_config.ini*, inside the folder *network_config_files/*:
```
python main.py -s minimum -t maximum -sd 10 -d ../network_config_files/network_config.ini
```

User-specified networks
-----------------------

For a custom network, you must specify the location of all the nodes in the same format used in the example file in the *network_ini_files/* folder. Moreover, if you would like to specify the SF and TP of all nodes, you can do so in the same format as in the example file provided in this repository. Additionally, you must enable the *-sno* flag when running the code. This flag indicates whether the SFs and TPs are in the network pickle file.

```
python main.py -s minimum -t maximum -sd 10 -d ../network_config_files/custom_network.ini -sno 1
```

Even though the *-s* and *-t* flags have no effect in this case, they must always be entered. Note that the format of the INI files are based on the config files used in the [FloRa simulator](https://github.com/florasim/flora).


Choosing which approach to run
==============================

You can choose whether to run the simulator with NoReL or ADR. To do this, use the option *-sm*, which can take the value of *NoReL* or *ADR*. The ADR implementation follows the specifications in [this paper](https://ieeexplore.ieee.org/document/8406255).


Dynamic scenarios
=================

There are three different dynamic scenarios supported by the simulator. These can be specified with the option *-ct*:

* *addNodes*: Adding new nodes to the network. The number of new nodes must be given with option *-nn*
* *channelConditions*: Changing the channel conditions as explained in [the paper](https://ieeexplore.ieee.org/document/9813414).
* *additionalTraffic*: Increasing the traffic in the network by a factor of 5.

The change is introduced in the middle of the simulation, e.g., if the simulation time is 10 days, then, the first 5 days are simulated with normal conditions and the last 5 days are simulated with the new (changed) conditions.


Saving your results
===================

You can store the final delivery ratio per node in a [numpy](https://numpy.org/) file, by specifying the path and file name you want to create with the results, using the option *-df*. Remember to include the file extension *.npy*. If such a file already exists, the results will be overwritten. Moreover, with option *-srf*, you can specify whether to calculate the delivery ratio as the ratio between received and transmit packets, as computed by the simulator, or as the delivery ratio given by [this model](https://github.com/VeronicaToro/LoRa-model). The default value is 1, which returns the delivery ratio as computed by the simulator. 
```
python main.py -s minimum -t maximum -sd 10 -d ../network_config_files/network_config.ini -df results.npy -srf 1
```

Moreover, you can choose to keep track of the overall network delivery ratio and distribution of SFs and TPs during the simulated time. To do this, turn up the option *-rt* and specify the output filename with option *-od*.

Changing parameters
===================

You would probably like to run the simulator with different channel and traffic parameters. To set the channel settings, you should modify the definition of *self._pld0*, *self._d0*, *self._gamma* and *self._sigma* in *src/network.py*. This simulator uses a [lognormal path loss](https://en.wikipedia.org/wiki/Log-distance_path_loss_model), where

* *self._pld0* is the mean pathloss at distance *self._d0*,
* *self._gamma* is the pathloss exponent and
* *self._sigma* is the standard deviation of a Gaussian variable with zero mean that models the channel variations.

For a different average sending rate, i.e., the average rate at which the LoRa devices send packets, you should set the variable *arrival_rate* in *src/main.py*.

The duty-cycle restriction can also be modified through the variable *self._dutyCycle* in *src/network.py*. A duty-cycle of 0.01 means that the devices can only be active for 1% of the time.
