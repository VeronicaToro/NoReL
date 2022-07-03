#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 08:18:20 2021

@author: Verónica Toro-Betancur and Gopika Premsankar
"""
import numpy as np
import math
import numpy
import logging

logger = logging.getLogger(__name__)

class Simulator:
    
    def __init__(self, parent, simulation_time, round_size, fixedTP, regret_learning_flag=False, ADR_flag=False, der_track_flag=False, statistics_flag=False):
        if regret_learning_flag * ADR_flag:
            logging.error("You cannot simulate NoReL and ADR at the same time")
            exit(1)
        self.parent = parent
        self._numNodes = self.parent._numNodes
        self._arrivalRate = self.parent._arrivalRate
        self._numGateways = self.parent._numGateways
        self._seed = self.parent._seedValue
        self.regret_learning_flag = regret_learning_flag
        self.adr_flag = ADR_flag
        self.der_track_flag = der_track_flag
        self.statistics_flag = statistics_flag
        self.der_track_time_interval = 60 * 60 # 1 hour
        self.record_results = {"DER":[], "num_packets":[], "SF":[], "TP":[]}
        self._pld0 = self.parent._pld0
        self._gamma = self.parent._gamma
        self._sigma = self.parent._sigma
        self._d0 = self.parent._d0
        self._calculated_pld0 = self._pld0
        self._calculated_sigma = self._sigma
        self._calculated_gamma = self._gamma
        self._calculated_lambda = self._arrivalRate
        self._num_packets_for_statistics = 100
        self._data_for_statistics = [[0]*self._num_packets_for_statistics, [0]]
        self._alpha = 0.3 # learning rate for statistics
        self.y_matrix = [0]*self._num_packets_for_statistics
        self.X_matrix = [[1]*self._num_packets_for_statistics, [0]*self._num_packets_for_statistics]
        self._transmission_times = dict()
        self._simulation_time = simulation_time
        self._dutyCycle = 0.01
        self._sensitivityReceiverPerSF = {7:-124, 8: -127, 9:-130, 10: -133, 11:-135, 12:-137}
        self._transmitTimes = {7: 0.07808, 8: 0.139776, 9: 0.246784, 10: 0.493568, 11: 0.856064, 12: 1.712128} # 20 B
        self._preambleTimes = {7:0.001024,	8:0.002048,	9:0.004096, 10:0.008192, 11:0.016384, 12:0.032768}
        self._spareSymbols = 7.25
        self._sirMatrix = [[1, -8, -9, -9, -9, -9],
                           [-11, 1, -11, -12, -13, -13],
                           [-15, -13, 1, -13, -14, -15],
                           [-19, -18, -17, 1, -17, -18],
                           [-22, -22, -21, -20, 1, -20],
                           [-25, -25, -25, -24, -23, 1]]
        self._round_size = round_size
        self._num_total_transmitted_packets_per_node = {x : 0 for x in range(self._numNodes)}
        self._num_total_received_packets_per_node = {x : 0 for x in range(self._numNodes)}
        self._missedPackets = {x : [] for x in range(self._numNodes)} #For statistics about reasons of dropping a packet: 0: outage, 1: collision 2:collision with downlink TX
        self._deliveryRatio = []
        self._successful_packets_per_round_per_node = {x : 0 for x in range(self._numNodes)}
        self._packets_sent_per_node = {x : 0 for x in range(self._numNodes)}
        self._previous_downlink_transmission_time = -100
        self._events_times = []
        self._sending_rate_events = [[self._simulation_time],[self._arrivalRate]]
        self._convergence_time = 0
        self._convergence_packets = 0
        if self.regret_learning_flag:
            self._DER_evolution_of_random_node = []
            self._done_with_node = dict()
            self.parent.regretLearning.calculate_strategies_per_node(fixedTP)
            self.parent.regretLearning.initialize_variables()
            for node in range(self._numNodes):
                self.parent.regretLearning._current_action[node] = (self.parent._nodes[node].get_SF(), self.parent._nodes[node].get_TP())
                self._done_with_node[node] = False
        if self.adr_flag:
            self._adr_ack_limit = 64
            self._adr_ack_delay = 32
            self._adr_ack_cnt = {x : 0 for x in range(self._numNodes)}
            self._SNR_last_packets = {x : [] for x in range(self._numNodes)}
            self._num_packets_for_adr_net = 20
            self._num_recvd_for_adr = [0 for x in range(self._numNodes)]
        if self.regret_learning_flag or self.adr_flag:
            for gw in range(self._numGateways):
                gateway_coords = self.parent.deployment._gateway_coords[gw]
                self.parent.deployment.add_nodes(1, passNodeCoords=True, nodeCoordinates=[gateway_coords], numOldNodes=self._numNodes)
            for gw1 in range(self._numGateways):
                gw1_coords = self.parent.deployment._gateway_coords[gw1]
                for gw2 in range(self._numGateways):
                    gw2_coords = self.parent.deployment._gateway_coords[gw2]
                    distToGW = np.sqrt(math.pow((gw1_coords[0]-gw2_coords[0]),2) + math.pow((gw1_coords[1]-gw2_coords[1]),2))
                    self.parent._nodes[self._numNodes + gw1].set_distance_to_gateway(distToGW, gw2)
        
    def calculate_tranmission_times_of_nodes(self):
        self._all_transmissions = [[],[]]
        for node in range(self._numNodes):
            self._transmission_times[node] = []
            sf = self.parent._nodes[node].get_SF()
            time = 0
            num_packets_for_node = 0
            if len(self._sending_rate_events[0]) > 1:
                idx = 1
            else: 
                idx = 0
            while time <= self._simulation_time:
                if time <= self._sending_rate_events[0][idx]:
                    time_interval = np.random.exponential(1 / self._sending_rate_events[1][idx - 1])
                    # Check if the next packet would be transmitted after the duty-cycle restriction
                    if time_interval >= (100 * (1 - self._dutyCycle) + 1) * self._transmitTimes[sf]:
                        time += time_interval
                        self._transmission_times[node].append(time)
                        num_packets_for_node += 1
                else:
                    idx += 1
            
            self._all_transmissions[0].extend([node] * num_packets_for_node)
            self._all_transmissions[1].extend(self._transmission_times[node])
        transposed_l = list(zip(*self._all_transmissions))
        transposed_l.sort(key=lambda x: x[1], reverse=False)
        self.sorted_transmission_times = list(zip(*transposed_l))
        self.sorted_transmission_times[0] = list(self.sorted_transmission_times[0])
        self.sorted_transmission_times[1] = list(self.sorted_transmission_times[1])
    
    def run_simulation(self, inform_nodes=False):
        logger.debug("Running simulation...")
        self.calculate_tranmission_times_of_nodes()
        self.num_total_transmissions = len(self.sorted_transmission_times[1])
        idx = -1
        idx_events = 0
        idx_statistics = 0
        if inform_nodes:
            nodes_to_inform = list(range(self._numNodes))
        num_sent_packets = 0
        num_sent_packets_per_node = [0] * self._numNodes
        for time in self.sorted_transmission_times[1]:
            if self._events_times:
                if time >= self._events_times[idx_events][0] and time < self._events_times[idx_events + 1][0]:
                    self.apply_event(idx_events)
                    idx_events += 1
            idx += 1
            success = False
            node = self.sorted_transmission_times[0][idx]
            if node < self._numNodes:
                num_sent_packets += 1
                num_sent_packets_per_node[node] += 1
                sf = self.parent._nodes[node].get_SF()
                tp = self.parent._nodes[node].get_TP()
                
                if inform_nodes and node in nodes_to_inform:
                    if time + self._transmitTimes[sf] + 2 >= self._previous_downlink_transmission_time + self._transmitTimes[12] * 100:
                        self.insert_downlink_transmission(node, time, idx)
                        nodes_to_inform.remove(node)
                        self._done_with_node[node] = False
                        self.parent.regretLearning.initialize_variables(nodes=[node])
                
                interferers = []
                # Check for interferer nodes that start transmission during the test node transmission
                idx2 = idx + 1
                if idx2 < len(self.sorted_transmission_times[0]):
                    init_time_interferer = self.sorted_transmission_times[1][idx2]
                    while init_time_interferer - time < self._transmitTimes[sf]:
                        interferers.append(self.sorted_transmission_times[0][idx2])
                        idx2 += 1
                        init_time_interferer = self.sorted_transmission_times[1][idx2]
                    
                # Check if the transmission of node overlaps with that of the interferer (after 7.25 preamble symbols)
                idx2 = idx - 1
                if idx2 >= 0:
                    init_time_interferer = self.sorted_transmission_times[1][idx2]
                    interferer = self.sorted_transmission_times[0][idx2]
                    sf_interferer = self.parent._nodes[interferer].get_SF()
                    while (init_time_interferer + self._transmitTimes[sf_interferer] > time + self._preambleTimes[sf] * self._spareSymbols) and (idx2 >= 0):
                        interferers.append(interferer)
                        idx2 -= 1
                        init_time_interferer = self.sorted_transmission_times[1][idx2]
                        interferer = self.sorted_transmission_times[0][idx2]
                        sf_interferer = self.parent._nodes[interferer].get_SF()
                
                # Decide which packets will be correctly decoded
                victories = []
                pathloss_per_GW = []
                interference_per_GW = []
                for gw in range(self._numGateways):
                    distance_node = self.parent._nodes[node].get_distance_to_gateway(gw)
                    pathloss_node = tp - self._pld0 - 10*self._gamma*math.log10(distance_node/self._d0) + np.random.normal(0, self._sigma, 1)[0]
                    pathloss_per_GW.append(pathloss_node)
                    interference_per_GW.append([])
                    # Check if the receive power is higher than the GW sensitivity
                    if pathloss_node >= self._sensitivityReceiverPerSF[sf]:
                        victory = True
                    else:
                        victory = False
                        self._missedPackets[node].append(0)
                        continue
                    for k in range(len(interferers)):
                        interferer = interferers[k]
                        sf_interferer = self.parent._nodes[interferer].get_SF()
                        tp_interferer = self.parent._nodes[interferer].get_TP()
                        distance_interferer = self.parent._nodes[interferer].get_distance_to_gateway(gw)
                        if distance_interferer:
                            pathloss_interferer = tp_interferer - self._pld0 - 10*self._gamma*math.log10(distance_interferer/self._d0) + np.random.normal(0, self._sigma, 1)[0]
                        else:
                            pathloss_interferer = tp_interferer
                        interference_per_GW[gw].append(pathloss_interferer)
                        # Check if there is capture effect or not with the interferer node
                        if pathloss_node - pathloss_interferer <= self._sirMatrix[sf - 7][sf_interferer - 7]:
                            victory = False
                            if interferer > self._numNodes: # If the interferer is a downlink transmission
                                self._missedPackets[node].append(2)
                            else:
                                self._missedPackets[node].append(1)
                            break
                        else:
                            victory = True
                    victories.append(victory)
                if any(victories):
                    self._num_total_received_packets_per_node[node] += 1
                    success = True
                
                if self.statistics_flag:
                    self.y_matrix.pop(0)
                    self.y_matrix.append(-pathloss_node + tp)
                    self.X_matrix[1].pop(0)
                    self.X_matrix[1].append(10.0 * math.log10(distance_node/self._d0))
                    y = pathloss_node - tp + self._pld0 + 10.0 * self._calculated_gamma * math.log10(distance_node/self._d0)
                    self._data_for_statistics[0].pop(0)
                    self._data_for_statistics[0].append(y)
                    idx_statistics += 1
                    if idx_statistics == self._num_packets_for_statistics:
                        X_matrix_aux = np.matrix(self.X_matrix)
                        aux1 = np.linalg.inv(np.matmul(X_matrix_aux, X_matrix_aux.T))
                        aux2 = np.matmul(X_matrix_aux, np.matrix(self.y_matrix).T)
                        aux = np.matmul(aux1, aux2)
                        self._calculated_pld0 = self._alpha * self._calculated_pld0 + (1 - self._alpha) * np.mean(aux[0])
                        self._calculated_gamma = self._alpha * self._calculated_gamma + (1 - self._alpha) * np.mean(aux[1])
                        self._calculated_sigma = self._alpha * self._calculated_sigma + (1 - self._alpha) * np.std(self._data_for_statistics[0])
                        self._calculated_lambda = (self._alpha) * self._calculated_lambda + (1 - self._alpha) * self._num_packets_for_statistics / (time - self._data_for_statistics[1][0]) / self._numNodes
                        self._data_for_statistics[1][0] = time
                        idx_statistics = 0
                    
                if self.regret_learning_flag:
                    self._packets_sent_per_node[node] += 1
                    if success:
                        self._successful_packets_per_round_per_node[node] += 1
                        if self._packets_sent_per_node[node] >= self._round_size:
                            self.send_observed_DER(node)
                            self.insert_downlink_transmission(node, time, idx)
                    if sum(self._done_with_node.values()) >= 0.95 * self._numNodes:
                        self._convergence_time = time
                        self._convergence_packets = num_sent_packets
#                        break
                    
                if self.adr_flag:
                    # First fill SNR packets received per node
                    if success:
                        self._num_recvd_for_adr[node] += 1
                        if len(self._SNR_last_packets[node]) == self._num_packets_for_adr_net:
                            self._SNR_last_packets[node].pop(0)
                        self._SNR_last_packets[node].append(
                            (max(pathloss_per_GW), interference_per_GW[np.argmax(pathloss_per_GW)]))

                    if (sf, tp) != (12, 14):
                        # ADR-NODE
                        self._adr_ack_cnt[node] += 1
                        if self._adr_ack_cnt[node] >= self._adr_ack_limit:
                            if success:
                                # Send ADR_ACK_REQ for packets sent after hitting ADR_ACK_LIMIT
                                downlink = self.parent.adr_baseline.ADR_NET(self._calculated_sigma, self._SNR_last_packets[node], node,
                                                                            force_ADR=True)
                                if downlink:
                                    self.insert_downlink_transmission(node, time, idx)
                                    self._adr_ack_cnt[node] = 0
                            if self._adr_ack_cnt[node] >= self._adr_ack_limit + self._adr_ack_delay and \
                                    (self._adr_ack_cnt[node] - self._adr_ack_limit) % self._adr_ack_delay == 0:
                                # ADR-NET command not received. Need to increase SF and TP
                                self.parent._nodes[node].set_TP(14)
                                self.parent._nodes[node].set_SF(min(sf + 1, 12))
                    if success and self._num_recvd_for_adr[node] == self._num_packets_for_adr_net:
                        # ADR-NET runs once 20 packets are received
                        downlink = self.parent.adr_baseline.ADR_NET(self._calculated_sigma, self._SNR_last_packets[node], node)
                        self._num_recvd_for_adr[node] = 0
                        if downlink:
                            self.insert_downlink_transmission(node, time, idx)
                            self._adr_ack_cnt[node] = 0

                self._num_total_transmitted_packets_per_node[node] += 1
            if self.der_track_flag:
                if time <= self.der_track_time_interval and self.sorted_transmission_times[1][min(idx + 1, self.num_total_transmissions - 1)] > self.der_track_time_interval:
                    self.parent.calculate_prob_outage()
                    self.parent.update_interferers_per_node()
                    model_deliveryRatio = self.parent.calculate_delivery_ratio_per_node(model="quasi-orthogonal-sigma", outputFile=None)
                    self.record_results["DER"].append(np.mean(model_deliveryRatio))
                    self.record_results["num_packets"].append(num_sent_packets)
                    SFs = [0] * 6
                    TPs = [0] * 5
                    for node in self.parent._nodes:
                        sf = node.get_SF()
                        tp = node.get_TP()
                        if sf > 6 and sf < 13 and tp > 1 and tp < 15:
                            SFs[sf - 7] += 1
                            TPs[int((tp - 2) / 3)] += 1
                    self.record_results["SF"].append(SFs)
                    self.record_results["TP"].append(TPs)
                    self.der_track_time_interval += 60 * 60
    
    def insert_downlink_transmission(self, node, time, idx):
        sf = self.parent._nodes[node].get_SF()
        # Determine closest GW that will send the observed utility to the node
        gw = np.argmin(list(self.parent._nodes[node].get_distance_to_all_gateways().values()))
        self.parent._nodes[self._numNodes + gw].set_SF(12)
        self.parent._nodes[self._numNodes + gw].set_TP(14)
        downlink_init = time + self._transmitTimes[sf] + 2 # TX in the second receive window of node
        time_loc = idx + 1
        if time_loc < self.num_total_transmissions:
            t = self.sorted_transmission_times[1][time_loc]
            while t < downlink_init:
                time_loc += 1
                t = self.sorted_transmission_times[1][time_loc]
            self.sorted_transmission_times[0].insert(time_loc, self._numNodes + gw)
            self.sorted_transmission_times[1].insert(time_loc, downlink_init)
            self.num_total_transmissions += 1
        self._previous_downlink_transmission_time = downlink_init
    
    def send_observed_DER(self, node):
        observed_utility = self._successful_packets_per_round_per_node[node] / self._round_size
        self._done_with_node[node] = self.parent.regretLearning.calculate_action(node, observed_utility, self._calculated_sigma, self._calculated_lambda)
        self._packets_sent_per_node[node] = 0
        self._successful_packets_per_round_per_node[node] = 0
    
    def calculate_avg_delivery_ratio(self):
        # This function gives the overall delivery ratio as calculated by the simulator
        sum_DER = 0
        num_received_pkts = 0
        num_sent_pkts = 0
        for node in range(self._numNodes):
            if self._num_total_transmitted_packets_per_node[node]:
                sum_DER += self._num_total_received_packets_per_node[node] / \
                           self._num_total_transmitted_packets_per_node[node] * 100
                num_received_pkts += self._num_total_received_packets_per_node[node]
                num_sent_pkts += self._num_total_transmitted_packets_per_node[node]
        return num_received_pkts / num_sent_pkts
    
    def apply_event(self, idx_events):
        # This function introduces changes in the pathloss parameters and traffic
        self.parent._pld0 = self._pld0 = self._events_times[idx_events][1]
        self.parent._gamma = self._gamma = self._events_times[idx_events][2]
        self.parent._sigma =  self._sigma = self._events_times[idx_events][3]
        self.parent._arrivalRate = self._arrivalRate = self._events_times[idx_events][4]
    
    def calculate_statistics(self, simulator_results_flag=True, destinationFile=""):
        if simulator_results_flag:
            # Delivery ratio as calculated by the simulator
            for node in range(self._numNodes):
                self._deliveryRatio.append(self._num_total_received_packets_per_node[node] / self._num_total_transmitted_packets_per_node[node] * 100)
        else:
            # Delivery ratio as given by the model in  https://ieeexplore.ieee.org/document/9488783
            self.parent.calculate_prob_outage()
            self.parent.update_interferers_per_node()
            self._deliveryRatio = self.parent.calculate_delivery_ratio_per_node(model="quasi-orthogonal-sigma", outputFile=None)
        
        if self.regret_learning_flag:
            print("How many nodes converged?", sum(self._done_with_node.values()))
            print("Convergence time:", self._convergence_time)
            print("Convergence packets:", self._convergence_packets)
            print("Mean delivery ratio:", np.mean(self._deliveryRatio))
            print("Max delivery ratio:", max(self._deliveryRatio))
            print("Min delivery ratio:", min(self._deliveryRatio))
            
        if destinationFile:
            np.save(destinationFile, self._deliveryRatio)