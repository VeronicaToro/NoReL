#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" This function calls classes to deploy and evaluate the network"""

import argparse
from network import Network
import logging.config
import configparser

logging.config.fileConfig('logging_config.ini')

parser = argparse.ArgumentParser(description='Evaluate delivery ratio per node in LoRa networks')
parser.add_argument('--sfStrategy', '-s', dest='sfStrategy', help='SF Strategy: random, minimum',
                    required=True)
parser.add_argument('--tpStrategy', '-t', dest='tpStrategy', help='TP Strategy: random, minimum, maximum',
                    required=True)
parser.add_argument('--seedValue', '-v', dest='seedValue', help='Seed value for the script',
                    required=False, type=int, default=42)
parser.add_argument('--directory', '-d', dest='directory', help='Directory where the config file is located', 
                    required=True, type=str)
parser.add_argument('--sfNotAssigned', '-sno', dest='sfNotAssigned', help='flag for whether the SFs and TPs are in the network pickle file (1=Yes, 0=No)',
                    required=False, type=int, default=0)
parser.add_argument('--destinationFile', '-df', dest='destinationFile', help='File name where the final delivery ratio per node is going to be saved', 
                    required=False, type=str, default=None)
parser.add_argument('--simMethod', '-sm', dest='simMethod', help='Method for the simulation: NoReL or ADR',
                    required=False, type=str, default=None)
parser.add_argument('--simulationDays', '-sd', dest='simulationDays', help='Number of days for simulation time',
                    required=True, default=1)
parser.add_argument('--newNodes', '-nn', dest='newNodes', help='Number of nodes that will be added to the network',
                    required=False, default=0)
parser.add_argument('--changesType', '-ct', dest='changesType', help='Type of change to be introduced in the network: addNodes, channelConditions or additionalTraffic',
                    required=False, type=str, default=None)
parser.add_argument('--results_track_flag', '-rt', dest='results_track_flag', help='Flag to save the evolution of DER, SF, TP in output files (1=Yes, 0=No)',
                    required=False, default=0)
parser.add_argument('--fixedTP', '-fTP', dest='fixedTP', help='Define TP to run NoReL with all nodes using the same TP',
                    required=False, default=0)
parser.add_argument('--simulator_results_flag', '-srf', dest='simulator_results_flag', help='Flag to show final delivery ratio given by the simulator (1=simulator, 0=model)',
                    required=False, default=1)
parser.add_argument('--outputDirectory', '-od', dest='outputDir', help='Directory where to locate the output files when results_track_flag has been set to 1', 
                    required=False, type=str)
args = parser.parse_args()

nwConfigParser = configparser.ConfigParser()
nwConfigParser.read(args.directory)
gatewayCoords = []
gatewayDeployRadius = []
numNodesPerGW = []
numNodes = 0
numGateways = 0
networkSizeX = 0
networkSizeY = 0
deployFromPickleFile = False
try:
    networkSizeX = float(nwConfigParser["network"]["sizeX"])
    networkSizeY = float(nwConfigParser["network"]["sizeY"])
    numNodes = int(nwConfigParser["network"]["numNodes"])
    numGateways = int(nwConfigParser["network"]["numGateways"])

    if nwConfigParser.has_option("network", "pickleFile"):
        # call a different deploy function
        deployFromPickleFile = True
        nwPickleFile = nwConfigParser["network"]["pickleFile"]
        logging.debug("Pickle file = {}".format(nwPickleFile))
    else:
        for gw_index in range(numGateways):
            section_name = "gateway_" + str(gw_index)
            xLocation = int(nwConfigParser[section_name]["xLocation"])
            yLocation = int(nwConfigParser[section_name]["yLocation"])
            gatewayCoords.append(tuple((xLocation, yLocation)))
            gatewayDeployRadius.append(float(nwConfigParser[section_name]["radius"]))
            numNodesPerGW.append(int(nwConfigParser[section_name]["numNodes"]))
        if numNodes != sum(numNodesPerGW):
            logging.error("Sum of nodes deployed around each gateway does not match total number of nodes")
            exit(1)
except configparser.ParsingError as err:
    logging.error("Error reading configuration parser = {}".format(err))
    
arrival_rate = 1 / 1000.0 # average sending rate of all nodes
network = Network(numNodes, numGateways, networkSizeX, networkSizeY, arrival_rate, args.seedValue)

if deployFromPickleFile == False:
    network.deploy_network(numNodes=numNodes, numGateways=numGateways, gatewayCoords=gatewayCoords,
                           deploymentArea="circle", nodeLocationType="random", circleRadius=gatewayDeployRadius,
                           numNodesPerGW=numNodesPerGW)
elif deployFromPickleFile == True:
    if args.sfNotAssigned == 1:
        # init network, and assign SFs and TPs from pickle
        network.deploy_network_from_pickle(numNodes=numNodes, numGateways=numGateways, pickleFileName=nwPickleFile)
    else:
        # Only initialize node and GW positions
        network.init_network_from_pickle(numNodes=numNodes, numGateways=numGateways, pickleFileName=nwPickleFile)

network.deployment.calculate_network_distance_parameters()

if not args.sfNotAssigned:
    network.configuration.set_possibleSFs_possibleTPs_allnodes()
    network.configuration.configure_TP_allnodes()
    network.configuration.configure_SF_allnodes()

network.network_simulator(args.simulationDays, args.simMethod, args.newNodes, args.changesType, args.results_track_flag, 
                          args.fixedTP, args.simulator_results_flag, args.outputDir, args.destinationFile)