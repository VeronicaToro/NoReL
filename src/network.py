#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
In this file, the model calculations take place

@author: Verónica Toro-Betancur and Gopika Premsankar
"""
import math
import numpy as np
import copy
import logging
import itertools
from deployment import Deployment
from configuration import Configuration
from file2data import file_reader
from collections import OrderedDict
from set_covers import calculate_setCovers
from simulator import Simulator
from ADR_implementation import ADR
from regret_learning import Regret_Learning

logger = logging.getLogger(__name__)

class Network(object):
    def __init__(self, numNodes, numGateways, networkSizeX, networkSizeY, arrivalRate, seedValue=1):
        self._numNodes = numNodes
        self._numGateways = numGateways
        self._networkSizeX = networkSizeX
        self._networkSizeY = networkSizeY
        self._arrivalRate = arrivalRate
        self._nodes = []
        self._gateways = []
        self._seedValue = seedValue
        
#        # Pathloss model for sub-urban environment taken from https://dl.acm.org/doi/10.1145/2988287.2989163
#        self._pld0 = 127.41
#        self._gamma = 2.08
#        self._sigma = 3.57
##        self._sigma = 0
#        self._d0 = 40
        
        # Pathloss model for sub-urban environment taken from https://doi.org/10.1109/ITST.2015.7377400
        self._pld0 = 128.95
        self._gamma = 2.32
        self._sigma = 7.08 / 2.0
        self._d0 = 1000
        
        self._dutyCycle = 0.01
        self._availableSFs = [7, 8, 9, 10, 11, 12]
        self._availableTPs = [2, 5, 8, 11, 14]
        # sensitivity values from Semtech datasheet
        self._sensitivityReceiverPerSF = {7:-124, 8: -127, 9:-130, 10: -133, 11:-135, 12:-137}
        self._transmitTimes = {7: 0.07808, 8: 0.139776, 9: 0.246784, 10: 0.493568, 11: 0.856064, 12: 1.712128} # for a payload of 20 B
#        self._transmitTimes = {7: 0.110848, 8: 0.205312, 9: 0.345088, 10: 0.690176, 11: 1.380352, 12: 2.49856} # for a payload of 35 B
        self._preambleTimes = {7:0.001024,	8:0.002048,	9:0.004096, 10:0.008192, 11:0.016384, 12:0.032768}
        self._spareSymbols = 7.25
        self._sirMatrix = [[1, -8, -9, -9, -9, -9],
                           [-11, 1, -11, -12, -13, -13],
                           [-15, -13, 1, -13, -14, -15],
                           [-19, -18, -17, 1, -17, -18],
                           [-22, -22, -21, -20, 1, -20],
                           [-25, -25, -25, -24, -23, 1]]
        
        self.deployment = Deployment(self)
        self.configuration = Configuration(self)
        
        self._maxCommDistancePerTP = OrderedDict()
        self._sfRangesPerTP = OrderedDict()
        self.set_max_distance_per_tp()
        
        self.regretLearning = Regret_Learning(self)
        
        
    def init_network_from_pickle(self, numNodes, numGateways, pickleFileName):

        logger.debug("Deploying network from file = {}".format(pickleFileName))
        distances, assignedSF, TPs, nodeCoords, gwCoords = file_reader(pickleFileName, read_SF_and_TP=False)
        assert len(gwCoords) == numGateways
        assert len(nodeCoords) == numNodes

        self.deployment.add_gateways(numGateways, gwCoords, self._maxCommDistancePerTP[14])
        self.deployment.add_nodes(numNodes, passNodeCoords=True, nodeCoordinates=nodeCoords)
        self.deployment.init_dist_vector_per_node(numGateways=numGateways,
                                                  gatewayCoords=gwCoords,
                                                  totalNodes=numNodes)

        self.set_pathloss_allnodes()

    def deploy_network_from_pickle(self, numNodes, numGateways, pickleFileName):
        logger.debug("Deploying network from file = {}".format(pickleFileName))
        distances, assignedSF, TPs, nodeCoords, gwCoords = file_reader(pickleFileName, read_SF_and_TP=True)

        self.deployment.add_gateways(numGateways, gwCoords, self._maxCommDistancePerTP[14])
        self.deployment.add_nodes(numNodes, passNodeCoords=True, nodeCoordinates=nodeCoords)
        self.deployment.init_dist_vector_per_node(numGateways=numGateways,
                                                  gatewayCoords=gwCoords,
                                                  totalNodes=numNodes)
        
        self.set_pathloss_allnodes()
        
        for i in range(self._numNodes):
            self._nodes[i].set_SF(assignedSF[i])
            self._nodes[i].set_TP(TPs[i])

    def deploy_network(self, numNodes, numGateways, gatewayCoords,circleRadius, numNodesPerGW,
                       deploymentArea, nodeLocationType,
                       nodesInRings=[], radii=[]):
        self.deployment.add_gateways(numGateways, gatewayCoords, self._maxCommDistancePerTP[14])
        self.deployment.add_nodes(numNodes)
        logger.debug("Deploying network with {gw} gateway(s) and {node} nodes in {area} with radius = {radius}".format(
            gw=numGateways,
            node=numNodes,
            area=deploymentArea,
            radius=circleRadius
        ))
        self.deployment.set_locations_allnodes(numGateways=numGateways,
                                               gatewayCoords=gatewayCoords,
                                               numNodesPerGW=numNodesPerGW,
                                               circleRadius=circleRadius,
                                               deploymentArea=deploymentArea,
                                               locationType=nodeLocationType,
                                               nodesInRings=nodesInRings,
                                               radii=radii)
        self.deployment.init_dist_vector_per_node(numGateways=numGateways,
                                                  gatewayCoords=gatewayCoords,
                                                  totalNodes=numNodes)
        self.set_pathloss_allnodes()
    
    def set_sf_ranges_alldist_alltp(self):
        """
        Sets dictionary _sfRangesPerTP for each available transmit power level
        Each item is a dictionary, with key=distance and value=set of SFs usable upto that distance
        For example, _sfRangesPerTP[2] = [(0, [7, 8, 9, 10, 11, 12]), (130, [8, 9, 10, 11, 12]), (181, [9, 10, 11, 12]), (251, [10, 11, 12]), (350, [11, 12]), (437, [12])]
        For txPower 2, between distance 0 to 129, SFs 7 to 12 can be used, between distance 130 to 180, SFs 8 to 12 can be used, and so on
        :return: None
        """
        for txPower in self._availableTPs:
            sfRanges = OrderedDict()
            for dist in np.arange(0.01, self._maxCommDistancePerTP[txPower], 0.01):
                subsetSFs = self.calculate_available_sfs_per_distance(dist, txPower)
                listOfKeys = list(sfRanges.keys())
                if listOfKeys:
                    lastKey = listOfKeys[-1]
                    if subsetSFs != sfRanges[lastKey]:
                        sfRanges[dist] = subsetSFs
                else:
                    sfRanges[0] = subsetSFs
            lastKey = listOfKeys[-1]
            sfRanges[self._maxCommDistancePerTP[txPower]] = sfRanges[lastKey]
            self._sfRangesPerTP[txPower] = sfRanges
    
    def set_max_distance_per_tp(self):
        for txPower in self._availableTPs:
            maxDistance = self._d0 * np.power(10,(txPower-self._pld0-self._sensitivityReceiverPerSF[max(self._availableSFs)])/(10 * self._gamma))
            self._maxCommDistancePerTP[txPower] = maxDistance
    
    def calculate_available_sfs_per_distance(self, distance, txPower):
        """
        :param distance:
        :param txPower:
        :return:list of spreading factors that can be used at a particular distance and with txPower
        """
        subsetSFs = []
        pathloss = self._pld0 + 10*self._gamma*math.log10(distance/self._d0) + self._sigma
        for sf in self._availableSFs:
            recvPower = txPower - pathloss
            if np.greater_equal(recvPower, self._sensitivityReceiverPerSF[sf]):
                subsetSFs.append(sf)
        return subsetSFs
        
    def set_pathloss_allnodes(self):
        max_pathloss = -2000
        max_node = -1
        counter = 0
        for n in range(self._numNodes):
            node = self._nodes[n]
            closest_gw_distance = 100000
            for gw in range(self._numGateways):
                distToGW = node.get_distance_to_gateway(gw)
                if distToGW < closest_gw_distance:
                    if distToGW == 0.0:
                        distToGW = 0.1
                        node.set_distance_to_gateway(distToGW,gw)
                    closest_gw_distance = distToGW
                pathloss = self._pld0 + 10*self._gamma*math.log10(distToGW/self._d0) + self._sigma
                if pathloss > max_pathloss and distToGW<=closest_gw_distance:
                    max_pathloss = pathloss
                    max_node = counter
                node.set_pathloss(pathloss, gw)
            counter += 1
        logger.debug("Maximum path loss = {} for node = {}".format(max_pathloss, max_node))
        
    def update_interferers_per_node(self, node=None, gw=None, interferer=None):
        """
        INPUTS  node (optional): Only the interferers of the specified nodes are calculated.
                                 Default: all the nodes in the network
                gw (optional): Only the interferers with respect to the specified gateways are calculated
                               Default: all the gateways in the network are considered
                interferer (optional): Only the speficied nodes are checked as potential interferers
                                       Default: All the other nodes in the network
        OUTPUT:     The interferers of each node are appended to the corresponding list in node.py
        DESCRIPTION:This function calculates the interferers for each node by comparing their receive
                    power with that of the node and using self._sirMatrix
        """
        if node != None:
            if type(node) == int:
                nodes = [node]
            else:
                nodes = node
        if gw != None:
            gateways = [gw]
        else:
            nodes = list(range(self._numNodes))
            gateways = list(range(self._numGateways))
        
        for i in nodes:
            self._nodes[i].init_setInterferersPerGW()
        for k in range(len(nodes)):
            i = nodes[k]
            nodeSF = self._nodes[i].get_SF()
            nodeTP = self._nodes[i].get_TP()
            for gw in gateways:
                nodeSIR = nodeTP - self._nodes[i].get_pathloss(gwnum=gw)
                if node != None:
                    nodesToCheck = list(range(self._numNodes))
                    nodesToCheck.remove(i)
                    if not nodesToCheck:
                        nodesToCheck =[]
                else:
                    nodesToCheck = list(range(i+1,self._numNodes))
                if interferer != None:
                    nodesToCheck = [interferer]
                for j in nodesToCheck:
                    interfererSF = self._nodes[j].get_SF()
                    interfererTP = self._nodes[j].get_TP()
                    interfererSIR = interfererTP - self._nodes[j].get_pathloss(gwnum=gw)
                    threshold = self._sirMatrix[nodeSF-7][interfererSF-7]
                    if self._sigma:
                        probability_of_interference = (1.0/2.0) * (math.erf((threshold-nodeSIR+interfererSIR)/(2*self._sigma*math.sqrt(2))) + 1)
                        distToGW = self._nodes[j].get_distance_to_gateway(gw)
                        interfererPathloss = self._nodes[j].get_TP() - self._pld0 - 10*self._gamma*math.log10(distToGW/self._d0)
                        p_min_interferer = self._sensitivityReceiverPerSF[interfererSF]
                        if (nodeSIR - interfererSIR < self._sirMatrix[nodeSF-7][interfererSF-7] + 2*self._sigma) and (interfererPathloss > p_min_interferer - 1*self._sigma):
                            self._nodes[i].append_interferer(j, interfererSF, gw)
                            self._nodes[i].sum_prob_interferer(probability_of_interference, j, interfererSF, gw)
                        
                        # Inverse case
                        threshold = self._sirMatrix[interfererSF-7][nodeSF-7]
                        probability_of_interference = (1.0/2.0) * (math.erf((threshold-interfererSIR+nodeSIR)/(2*self._sigma*math.sqrt(2))) + 1)
                        distToGW = self._nodes[i].get_distance_to_gateway(gw)
                        interfererPathloss = self._nodes[i].get_TP() - self._pld0 - 10*self._gamma*math.log10(distToGW/self._d0)
                        p_min_interferer = self._sensitivityReceiverPerSF[nodeSF]
                        if (interfererSIR - nodeSIR < self._sirMatrix[interfererSF-7][nodeSF-7] + 2*self._sigma) and (interfererPathloss > p_min_interferer - 1*self._sigma):
                            self._nodes[j].append_interferer(i, nodeSF, gw)
                            self._nodes[j].sum_prob_interferer(probability_of_interference, i, nodeSF, gw)
                            
                    else:
                        if (nodeSIR - interfererSIR < self._sirMatrix[nodeSF-7][interfererSF-7]) and not (self._nodes[j].get_prob_outage()[gw]):
                            self._nodes[i].append_interferer(j, interfererSF, gw)
                            self._nodes[i].sum_prob_interferer(1, j, interfererSF, gw)
                        
                        # Inverse case
                        if (interfererSIR - nodeSIR < self._sirMatrix[interfererSF-7][nodeSF-7]) and not (self._nodes[i].get_prob_outage()[gw]):
                            self._nodes[j].append_interferer(i, nodeSF, gw)
                            self._nodes[j].sum_prob_interferer(1, i, nodeSF, gw)
    
    def calculate_prob_outage(self, node=None, gw=None):
        """
        INPUTS  node (optional): Only the probability of outage of the specified nodes are calculated
                                 Default: all the nodes in the network
                gw (optional): Only the probability of outage with respect to the specified gateways are calculated
                               Default: all the gateways in the network are considered
        OUTPUT:     The probability of outage of each node per gateway is appended to the corresponding list in node.py
        DESCRIPTION:This function calculates the probability of outage for each node and with respect to each gateway.
                    This is done by comparing the receive power of the node with the sensitivity of the gateway at the current
                    SF of the node.
        """
        if node != None and gw != None:
            nodes = [node]
            gateways = [gw]
        else:
            nodes = list(range(self._numNodes))
            gateways = list(range(self._numGateways))
        if not self._sigma:
            for node in nodes:
                for gw in gateways:
                    sf = self._nodes[node].get_SF()
                    distToGW = self._nodes[node].get_distance_to_gateway(gw)
                    pathloss = self._nodes[node].get_TP() - self._pld0 - 10*self._gamma*math.log10(distToGW/self._d0)
                    p_min = self._sensitivityReceiverPerSF[sf]
                    if pathloss >= p_min:
                        self._nodes[node].set_prob_outage(0, gw)
                    else:
                        self._nodes[node].set_prob_outage(1, gw)
        else:
            for node in nodes:
                for gw in gateways:
                    sf = self._nodes[node].get_SF()
                    distToGW = self._nodes[node].get_distance_to_gateway(gw)
                    pathloss = self._nodes[node].get_TP() - self._pld0 - 10*self._gamma*math.log10(distToGW/self._d0)
                    p_min = self._sensitivityReceiverPerSF[sf]
                    OutageProb = 1 - 1/(2) * (math.erf((pathloss - p_min)/(math.sqrt(2)*self._sigma)) + 1)
                    self._nodes[node].set_prob_outage(OutageProb, gw)
    
    
    def calculate_setCovers_of_gateways(self, listOfGWs):
        combinationsOfGWs = dict()
        combinationsOfGWs2 = []
        idx = 0
        for gw in range(1, self._numGateways+1):
            combinationsOfGWs2.append(list(itertools.combinations(listOfGWs, gw)))
            for seti in combinationsOfGWs2[-1]:
                combinationsOfGWs[idx] = set(seti)
                idx += 1
        setCovers = calculate_setCovers(set(listOfGWs), combinationsOfGWs)
        return setCovers, combinationsOfGWs2
    
    def calculate_setCovers_forAll_GW_combinations(self):
        listOfGWs = list(np.arange(self._numGateways))
        GWs_combinations = []
        GWsetCovers_dict = dict()
        combinationsOfGWs_dict = dict()
        for gw in range(1, self._numGateways+1):
            GWs_combinations.append(list(itertools.combinations(listOfGWs, gw)))
        for first_level in GWs_combinations[0]:
            for tupleOfGWs in first_level:
                tup = tuple([tupleOfGWs])
                GWsetCovers_dict[tup] = [[tup]]
                combinationsOfGWs_dict[tup] = [[tup]]
        for first_level in GWs_combinations[1:]:
            for tupleOfGWs in first_level:
                GWsetCovers_dict[tupleOfGWs],  combinationsOfGWs_dict[tupleOfGWs] = self.calculate_setCovers_of_gateways(list(tupleOfGWs))
        return GWsetCovers_dict,  combinationsOfGWs_dict
    
    
    def calculate_delivery_ratio_per_node(self, model="quasi-orthogonal-noSigma", outputFile=None, node=None):
        """
        INPUTS  model: Name of the model to be calculated. The possible values are:
                    quasi-orthogonal-noSigma (default): Calculates the model with no channel variations
                    quasi-orthogonal-sigma: Calculates the model with channel variations given by self._sigma
                outputFile (optional): Specifies the name of the npy file where the delivery ratio per node will be saved 
                                       Default: No file is created
                node (optional): Only the delivery ratio of the specified nodes are calculated
                                 Default: All the nodes in the network
        OUTPUT:     Delivery ratio per node in the numpy array self.deliveryRatio
        DESCRIPTION:This function calculates the delivery ratio per node according to the formulation in 
                    https://doi.org/10.1109/INFOCOM42981.2021.9488783
        """
        if not self._sigma:
            model = "quasi-orthogonal-noSigma"
        if node != None:
            if type(node) == int:
                nodes = [node]
            else:
                nodes = node
        else:
            nodes = list(range(self._numNodes))
        interferers = dict()
        nodes_per_SF = dict()
        self.deliveryRatio = np.zeros(self._numNodes)
        for sf in self._availableSFs:
            nodes_per_SF[sf] = []
        for i in nodes:
            sf = self._nodes[i].get_SF()
            nodes_per_SF[sf].append(i)
        GWsetCovers_dict,  combinationsOfGWs_dict = self.calculate_setCovers_forAll_GW_combinations()
        for i in nodes:
            sf = self._nodes[i].get_SF()
            if model == "quasi-orthogonal-noSigma":
               nodeSF = self._nodes[i].get_SF()
               Prob_of_failure = 0
               listOfGWs = list(range(self._numGateways))
               InterferersPerGW = []
               interferers[i] = self._nodes[i].get_prob_interferers_list_PerGW()
               for gw in range(self._numGateways):
                   if self._nodes[i].get_prob_outage()[gw]:
                       listOfGWs.remove(gw)
                       InterferersPerGW.append(set())
                   else:
                       InterferersPerGW.append(set(interferers[i][gw]))
               if not listOfGWs:
                   self.deliveryRatio[i] = 0
               else:
                   GWsetCovers = GWsetCovers_dict[tuple(listOfGWs)]
                   combinationsOfGWs = combinationsOfGWs_dict[tuple(listOfGWs)].copy()
                   sharedInPairs = dict()
                   combinationsOfGWs.reverse()
                   accumulatedNodes = set()
                   for line in combinationsOfGWs:
                       for tup in line:
                           allNodes = []
                           for gw in tup:
                               allNodes.append(set(InterferersPerGW[gw]))
                           currentSet = set.intersection(*allNodes)
                           sharedInPairs[tuple(tup)] = currentSet - accumulatedNodes
                           if len(tuple(tup)) > 1:
                               accumulatedNodes.update(currentSet)
                       
                   OrTerms = []
                   
                   for pairOfGWs in GWsetCovers:
                       other_probs3 = 1
                       for d in range(len(pairOfGWs)):
                           InterferersPerSF = {7:[], 8:[], 9:[], 10:[], 11:[], 12:[]}
                           if type(pairOfGWs[d]) == np.int64:
                               for k in sharedInPairs[tuple([pairOfGWs[d]])]:
                                   InterferersPerSF[self._nodes[k].get_SF()].append(k)
                           else:
                               for k in sharedInPairs[tuple(pairOfGWs[d])]:
                                   InterferersPerSF[self._nodes[k].get_SF()].append(k)
                           other_probs2 = 1
                           for s in self._availableSFs:
                               other_probs2 *= math.exp(-1  * (self._transmitTimes[s] + self._transmitTimes[nodeSF] - self._preambleTimes[nodeSF]*self._spareSymbols) * self._arrivalRate * len(InterferersPerSF[s]) * (1-(100*(1-self._dutyCycle))*self._transmitTimes[s] * self._arrivalRate))
                           
                           other_probs3 *= 1 - other_probs2
                       Prob_of_failure += other_probs3
                       OrTerms.append(other_probs3)
                   
                   
                   if len(OrTerms) > 1:
                       x = [element for element in OrTerms]
                       M = np.sum(x)
                       sum2 = 0
                       sums = []
                       for f in x:
                           M -= f
                           sum2 += f*M
                           sums.append(f*M)
                       probTerms = [sum2]
                       for h in range(len(OrTerms)-1):
                           sum3 = 0
                           sums2 = []
                           for k in range(1, len(sums)+1):
                               sum3 += x[k-1]*np.sum(sums[k:])
                               sums2.append(x[k-1]*np.sum(sums[k:]))
                           probTerms.append(sum3)
                           sums = sums2.copy()
                       
                       aux = 0
                       exponent = 1
                       for term in probTerms:
                           aux += (-1)**exponent * term
                           exponent += 1
                                   
                       Prob_of_failure += aux
                   
                   probOutageList = [self._nodes[i].get_prob_outage()[gw] for gw in listOfGWs]
                   probOutage = np.prod(probOutageList)
                   self.deliveryRatio[i] = (1 - Prob_of_failure)*(1 - probOutage)
                           
            elif model == "quasi-orthogonal-sigma":
               nodeSF = self._nodes[i].get_SF()
               Prob_of_failure = 0
               listOfGWs = list(range(self._numGateways))
               InterferersPerGW = []
               interferers[i] = self._nodes[i].get_prob_interferers_list_PerGW()
               for gw in range(self._numGateways):
                   distToGW = self._nodes[i].get_distance_to_gateway(gw)
                   pathloss = self._nodes[i].get_TP() - self._pld0 - 10*self._gamma*math.log10(distToGW/self._d0)
                   p_min = self._sensitivityReceiverPerSF[sf]
                   if pathloss < p_min - self._sigma:
                       listOfGWs.remove(gw)
                       InterferersPerGW.append(set())
                   else:
                       InterferersPerGW.append(set(interferers[i][gw]))
               if not listOfGWs:
                   self.deliveryRatio[i] = 0
               else:
                   GWsetCovers = GWsetCovers_dict[tuple(listOfGWs)].copy()
                   combinationsOfGWs = combinationsOfGWs_dict[tuple(listOfGWs)].copy()
                   sharedInPairs = dict()
                   combinationsOfGWs.reverse()
                   accumulatedNodes = set()
                   for line in combinationsOfGWs:
                       for tup in line:
                           allNodes = []
                           for gw in tup:
                               allNodes.append(set(InterferersPerGW[gw]))
                           currentSet = set.intersection(*allNodes)
                           sharedInPairs[tuple(tup)] = currentSet #- accumulatedNodes
                           if len(tuple(tup)) > 1:
                               accumulatedNodes.update(currentSet)
                       
                   OrTerms = []
                   
                   for pairOfGWs in GWsetCovers:
                       sumOverRegion = 1
                       sharedCopy = copy.deepcopy(sharedInPairs)
                       for d in range(len(pairOfGWs)):
                           InterferersPerSF = {7:[], 8:[], 9:[], 10:[], 11:[], 12:[]}
                           for k in sharedCopy[tuple(pairOfGWs[d])]:
                               InterferersPerSF[self._nodes[k].get_SF()].append(k)
                           other_probs2 = 1
                           for s in self._availableSFs:
                               other_probs = 1
                               for node in InterferersPerSF[s]:
                                   allProbs = []
                                   for g in pairOfGWs[d]:
                                       if pairOfGWs != [(0,), (1,)]:
                                           if node in interferers[i][g]:
                                               allProbs.append(interferers[i][g][node])
                                   listOfTheRestOfGWs = list(set(listOfGWs) - set(pairOfGWs[d]))
                                   for g in listOfTheRestOfGWs:
                                       if node in interferers[i][g]:
                                           allProbs.append(1 - interferers[i][g][node])
                                   aux = np.prod(allProbs) 
                                   other_probs *= 1 - aux * (1 - math.exp(-1  * (self._transmitTimes[s] + self._transmitTimes[nodeSF] - self._preambleTimes[nodeSF]*self._spareSymbols) * self._arrivalRate * (1-(100*(1-self._dutyCycle))*self._transmitTimes[s] * self._arrivalRate)))
                               other_probs2 *= other_probs 
                           
                           other_probs2 = 1 - other_probs2
                           
                           sumOverRegion *= other_probs2
                       
                       Prob_of_failure += sumOverRegion
                       OrTerms.append(sumOverRegion)
                   
                   
                   if len(OrTerms) > 1:
                       x = [element for element in OrTerms]
                       M = np.sum(x)
                       sum2 = 0
                       sums = []
                       for f in x:
                           M -= f
                           sum2 += f*M
                           sums.append(f*M)
                       probTerms = [sum2]
                       for h in range(len(OrTerms)-1):
                           sum3 = 0
                           sums2 = []
                           for k in range(1, len(sums)+1):
                               sum3 += x[k-1]*np.sum(sums[k:])
                               sums2.append(x[k-1]*np.sum(sums[k:]))
                           probTerms.append(sum3)
                           sums = sums2.copy()
                       
                       aux = 0
                       exponent = 1
                       for term in probTerms:
                           aux += (-1)**exponent * term
                           exponent += 1
                                   
                       Prob_of_failure += aux
                   
                   probOutageList = [self._nodes[i].get_prob_outage()[gw] for gw in listOfGWs]
                   probOutage = np.prod(probOutageList)
                   self.deliveryRatio[i] = (1 - Prob_of_failure)*(1 - probOutage)
            
        self.deliveryRatio = self.deliveryRatio * 100
        if outputFile is not None:
            logging.debug("Saving DER per node to npy file {}".format(outputFile))
            np.save(outputFile, self.deliveryRatio)
        return self.deliveryRatio
    
    def calculate_available_SFs_and_TPs_per_node(self):
        distance_nodes_dict = dict()
        for n in range(self._numNodes):
            node = self._nodes[n]
            distances = node.get_distance_to_all_gateways()
            distance_to_closestGW = min(list(distances.values()))
            distance_nodes_dict[node.get_id()] = distance_to_closestGW
            
            availableSFs = self._availableSFs.copy()
            availableTPs = {sf:[] for sf in availableSFs}
            for txPower in self._availableTPs:
                aux = self.calculate_available_sfs_per_distance(distance_to_closestGW, txPower)
                for sf in aux:
                    availableTPs[sf].append(txPower)
            for sf in self._availableSFs:
                if not availableTPs[sf]:
                    availableSFs.remove(sf)
                    del availableTPs[sf]
            if not availableSFs:
                availableSFs = [12]
            if not availableTPs:
                availableTPs[12] = [14]
            node.set_available_SFs(availableSFs)
            node.set_available_TPs(availableTPs)
        self.nodes_sorted_by_distance = [k for k,v in sorted(distance_nodes_dict.items(), key=lambda item: item[1])]
        
    
    def network_simulator(self, simulation_days, simMethod, new_nodes=0, changes_type=None, results_track_flag=False, 
                          fixedTP_flag=None, simulator_results_flag=True, filename="", destinationFile=""):
        
        simulation_time = float(simulation_days) * 24 * 3600
        statistics_flag=True
        round_size = 10
        if fixedTP_flag:
            fixedTP = fixedTP_flag
        else:
            fixedTP = 0
        
        if simMethod == "NoReL":
            RL_flag = True
            ADR_flag = False
        if simMethod == "ADR":
            RL_flag = False
            ADR_flag = True
            self.adr_baseline = ADR(self)
        
        if not changes_type:
            self.simulator = Simulator(self, simulation_time, round_size, fixedTP, regret_learning_flag=RL_flag, ADR_flag=ADR_flag, der_track_flag=results_track_flag, statistics_flag=statistics_flag)
            self.simulator.run_simulation()

        if changes_type == "addNodes":
            ## Running the first half of the simulation with the initial number of nodes
            self.simulator = Simulator(self, simulation_time / 2.0, round_size, fixedTP, regret_learning_flag=RL_flag, ADR_flag=ADR_flag, der_track_flag=results_track_flag, statistics_flag=statistics_flag)
            self.simulator.run_simulation()
            
            ## Introduce new nodes
            simulation_time /= 2.0 # Simulation time after adding the new nodes
            self.deployment.add_nodes(new_nodes, passNodeCoords=False, nodeCoordinates=[], numOldNodes=self._numNodes)
                
            for gw in range(self._numGateways):
                gw_coords = self.deployment._gateway_coords[gw]
                for node in range(self._numNodes, self._numNodes + new_nodes):
                    xLocation, yLocation = self.deployment.get_coordinates_for_node(deploymentType="circle", locationInitialization="random",
                                         radius=self.radius, gatewayCoords=gw_coords)
                    self.deployment.set_location_for_node(node, xLocation, yLocation)
                    distToGW = np.sqrt(math.pow((xLocation-gw_coords[0]),2) + math.pow((yLocation-gw_coords[1]),2))
                    self._nodes[node].set_distance_to_gateway(distToGW, gw)
                
            self._numNodes = self._numNodes + new_nodes
            self.calculate_available_SFs_and_TPs_per_node()
            self.set_pathloss_allnodes()
            for node in range(self._numNodes - new_nodes, self._numNodes):
                all_SFs = self._nodes[node].get_available_SFs()
                sf = min(all_SFs)
                tp = 14
                self._nodes[node].set_SF(sf)
                self._nodes[node].set_TP(tp)
                
            
            self.simulator._numNodes = self._numNodes
            self.simulator.der_track_time_interval = 60 * 60 # 1 hour
            self.simulator._transmission_times = dict()
            self.simulator._simulation_time = simulation_time
            self.simulator._previous_downlink_transmission_time = -100
            self.simulator._sending_rate_events = [[simulation_time],[self._arrivalRate]]
            for node in range(self._numNodes - new_nodes, self._numNodes):
                self.simulator._num_total_transmitted_packets_per_node[node] = 0
                self.simulator._num_total_received_packets_per_node[node] = 0
                self.simulator._missedPackets[node] = []
                self.simulator._successful_packets_per_round_per_node[node] = 0
                self.simulator._packets_sent_per_node[node] = 0
                    
            if RL_flag:
                self.regretLearning.calculate_strategies_per_node()
                self.regretLearning.initialize_variables(nodes=list(range(self._numNodes - new_nodes, self._numNodes)))
                self.simulator._convergence_time = 0
                self.simulator._convergence_packets = 0
                for node in range(self._numNodes - new_nodes, self._numNodes):
                    self.regretLearning._current_action[node] = (self._nodes[node].get_SF(), self._nodes[node].get_TP())
                    self.simulator._done_with_node[node] = False
            
            if ADR_flag:
                for node in range(self._numNodes - new_nodes, self._numNodes):
                    self.simulator._adr_ack_cnt[node] = 0
                    self.simulator._SNR_last_packets[node] = []
                    self.simulator._num_recvd_for_adr.extend([0] * new_nodes)
                        
            ## Re-introducing the GWs
            for gw in range(self._numGateways):
                gateway_coords = self.deployment._gateway_coords[gw]
                self.deployment.add_nodes(1, passNodeCoords=True, nodeCoordinates=[gateway_coords], numOldNodes=self._numNodes)
            for gw1 in range(self._numGateways):
                gw1_coords = self.deployment._gateway_coords[gw1]
                for gw2 in range(self._numGateways):
                    gw2_coords = self.deployment._gateway_coords[gw2]
                    distToGW = np.sqrt(math.pow((gw1_coords[0]-gw2_coords[0]),2) + math.pow((gw1_coords[1]-gw2_coords[1]),2))
                    self._nodes[self._numNodes + gw1].set_distance_to_gateway(distToGW, gw2)
            self.simulator.calculate_tranmission_times_of_nodes()
            self.simulator.run_simulation()
            
            
        if changes_type == "channelConditions":
            event_times = dict()
            time = 0
            for u in range(2):
                if not u % 2:
                    ## Initial channel conditions
                    pld0 = self._pld0
                    gamma = self._gamma
                    sigma = self._sigma
        
                    event_times[u] = [time, pld0, gamma, sigma, self._arrivalRate]
                    time += simulation_time / 2.0
                else:
                    ## Change channel conditions
                    pld0 = self._pld0
                    gamma += 1
                    sigma *= 2
        
                    event_times[u] = [time, pld0, gamma, sigma, self._arrivalRate]
                    time += simulation_time / 2.0           
            
            event_times[u+1] = [float("inf"), pld0, gamma, sigma, self._arrivalRate]
        
            self.simulator = Simulator(self, simulation_time, round_size, fixedTP, regret_learning_flag=RL_flag, ADR_flag=ADR_flag, der_track_flag=results_track_flag, statistics_flag=statistics_flag)
            self.simulator._events_times = event_times
            self.simulator.run_simulation()
            
        if changes_type == "additionalTraffic":
            event_times = dict()
            time = 0
            for u in range(2):
                if not u % 2:
                    ## Initial sending rate
                    arrivalRate = self._arrivalRate
        
                    event_times[u] = [time, self._pld0, self._gamma, self._sigma, arrivalRate]
                    time += simulation_time / 2.0
                else:
                    ## Change sending rate
                    arrivalRate *= 5
        
                    event_times[u] = [time, self._pld0, self._gamma, self._sigma, arrivalRate]
                    time += simulation_time / 2.0          
            
            event_times[u+1] = [float("inf"), self._pld0, self._gamma, self._sigma, arrivalRate]
        
            self.simulator = Simulator(self, simulation_time, round_size, fixedTP, regret_learning_flag=RL_flag, ADR_flag=ADR_flag, der_track_flag=results_track_flag, statistics_flag=statistics_flag)
            self.simulator._events_times = event_times
            self.simulator._sending_rate_events = [[event_times[x][0] for x in event_times], [event_times[x][4] for x in event_times]]
            self.simulator.run_simulation()
            
        
        self.simulator.calculate_statistics(simulator_results_flag=simulator_results_flag, destinationFile=destinationFile)
        if results_track_flag:
            self.save_npy_files(self.simulator.record_results, filename=filename)
            


    def save_npy_files(self, data, filename="results"):
        np.save(filename + "/output-recorded_DER-seed" + str(self._seedValue) + ".npy", data["DER"])
        np.save(filename + "/output-recorded_Packets-seed" + str(self._seedValue) + ".npy", data["num_packets"])
        np.save(filename + "/output-recorded_SF-seed" + str(self._seedValue) + ".npy", data["SF"])
        np.save(filename + "/output-recorded_TP-seed" + str(self._seedValue) + ".npy", data["TP"])