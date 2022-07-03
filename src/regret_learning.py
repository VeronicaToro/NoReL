#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 13:37:00 2021

@author: Verónica Toro-Betancur
"""
import numpy as np
import math

class Regret_Learning:
    
    def __init__(self, parent):
        self.parent = parent
        self._transmission_times = dict()
        self._rounds_per_node = dict()
        self._delivery_ratio_in_time = dict()
        self._SFs_in_time = dict()
        self._TPs_in_time = dict()
        self._set_of_actions = dict()
        self._current_action = dict()
        self._time_vector = dict()
        self._round_size = 10
        
        self._utility = dict()
        self._regret = dict()
        self._pi = dict()
        self._beta = dict()
        
        self._nu = dict()
        self._gamma = dict()
        self._xi = dict()
        self._mu = dict()
        self._kappa = dict()
        self._phi = dict()
        
        self.Learning_rate_exp = 0.8
        
        self._time_slot_per_node = dict()
        
        self._number_of_channel_types = 4
        self._number_of_traffic_types = 2
        self._max_channel_variance = 20
        self._max_traffic_intensity = 0.007
        self._channel_type_step = self._max_channel_variance / self._number_of_channel_types
        self._traffic_type_step = self._max_traffic_intensity / self._number_of_traffic_types
    
    def calculate_strategies_per_node(self, fixedTP=0):
        self.parent.calculate_available_SFs_and_TPs_per_node()
        for node in range(self.parent._numNodes):
            self.parent._nodes[node].calculate_available_strategies(fixedTP)
            self._set_of_actions[node]  = self.parent._nodes[node].get_strategies()
                
    def initialize_variables(self, nodes=[]):
        if not nodes:
            nodes = list(range(self.parent._numNodes))
            
            for ct in range(self._number_of_channel_types):
                self._nu[ct] = dict()
                self._gamma[ct] = dict()
                self._xi[ct] = dict()
                self._mu[ct] = dict()
                self._kappa[ct] = dict()
                
                self._utility[ct] = dict()
                self._regret[ct] = dict()
                self._pi[ct] = dict()
                self._beta[ct] = dict()
                
                self._time_slot_per_node[ct] = dict()
                
                for tt in range(self._number_of_traffic_types):
                    self._nu[ct][tt] = dict()
                    self._gamma[ct][tt] = dict()
                    self._xi[ct][tt] = dict()
                    self._mu[ct][tt] = dict()
                    self._kappa[ct][tt] = dict()
                    
                    self._utility[ct][tt] = dict()
                    self._regret[ct][tt] = dict()
                    self._pi[ct][tt] = dict()
                    self._beta[ct][tt] = dict()
                    
                    self._time_slot_per_node[ct][tt] = dict()
                
                    for node in nodes:
                        self._nu[ct][tt][node] = 1
                        self._gamma[ct][tt][node] = 1
                        self._xi[ct][tt][node] = 1
                        self._mu[ct][tt][node] = 1
                        self._kappa[ct][tt][node] = 0
                        
                        self._utility[ct][tt][node] = dict()
                        self._regret[ct][tt][node] = dict()
                        self._pi[ct][tt][node] = dict()
                        self._beta[ct][tt][node] = dict()
                        
                        self._time_slot_per_node[ct][tt][node] = 1
                        
                        # Initialize all vectors to zero
                        for action in self._set_of_actions[node]:
                            self._utility[ct][tt][node][action] = 0
                            self._regret[ct][tt][node][action] = 0
                            self._pi[ct][tt][node][action] = 0
                            self._beta[ct][tt][node][action] = 0
        if nodes:
            for ct in range(self._number_of_channel_types):
                for tt in range(self._number_of_traffic_types):
                    for node in nodes:
                        self._nu[ct][tt][node] = 1
                        self._gamma[ct][tt][node] = 1
                        self._xi[ct][tt][node] = 1
                        self._mu[ct][tt][node] = 1
                        self._kappa[ct][tt][node] = 0
                        
                        self._utility[ct][tt][node] = dict()
                        self._regret[ct][tt][node] = dict()
                        self._pi[ct][tt][node] = dict()
                        self._beta[ct][tt][node] = dict()
                        
                        self._time_slot_per_node[ct][tt][node] = 1
                        
                        # Initialize all vectors to zero
                        for action in self._set_of_actions[node]:
                            self._utility[ct][tt][node][action] = 0
                            self._regret[ct][tt][node][action] = 0
                            self._pi[ct][tt][node][action] = 0
                            self._beta[ct][tt][node][action] = 0
        
                
    def calculate_action(self, node, observed_utility, channel_state, traffic_state):        
        ct = min(self._number_of_channel_types - 1, int(channel_state // self._channel_type_step))
        tt = min(self._number_of_traffic_types - 1, int(traffic_state // self._traffic_type_step))
        self._kappa[ct][tt][node] += self._time_slot_per_node[ct][tt][node]**2
        
        # Update learning rates
        self._nu[ct][tt][node] = 1 / self._time_slot_per_node[ct][tt][node]**self.Learning_rate_exp
        self._gamma[ct][tt][node] = 1 / self._time_slot_per_node[ct][tt][node]**(self.Learning_rate_exp + 0.1)
        self._mu[ct][tt][node] = 1 / self._time_slot_per_node[ct][tt][node]**(self.Learning_rate_exp + 0.2)
        
        # Update utility and regret
        for action in self._set_of_actions[node]:
            if self._current_action[node] == action:
                prev_u = self._utility[ct][tt][node][action]
                self._utility[ct][tt][node][action] += self._nu[ct][tt][node] * (observed_utility - prev_u)
            prev_r = self._regret[ct][tt][node][action]
            self._regret[ct][tt][node][action] += self._gamma[ct][tt][node] * (self._utility[ct][tt][node][action] - prev_r - observed_utility)
            
        # Update denominator in beta
        denominator_beta = 0
        for action in self._set_of_actions[node]:
            if self._kappa[ct][tt][node] * max(0, self._regret[ct][tt][node][action]) > 709:
                denominator_beta = float("inf")
            else:
                denominator_beta += math.exp(self._kappa[ct][tt][node] * max(0, self._regret[ct][tt][node][action]))
        
        # Update beta and pi
        infinity_numerators = []
        index_infinity_numerators = []
        for action in self._set_of_actions[node]:
            if self._kappa[ct][tt][node] * max(0, self._regret[ct][tt][node][action]) > 709:
                numerator_beta = float("inf")
                infinity_numerators.append(self._kappa[ct][tt][node] * max(0, self._regret[ct][tt][node][action]))
                index_infinity_numerators.append(action)
            else:
                numerator_beta = math.exp(self._kappa[ct][tt][node] * max(0, self._regret[ct][tt][node][action]))
            if numerator_beta == float("inf") and denominator_beta == float("inf"):
                self._beta[ct][tt][node][action] = 1
            else:
                self._beta[ct][tt][node][action] = numerator_beta / denominator_beta
        sum_inf_numerators = sum(infinity_numerators)
        for a in range(len(index_infinity_numerators)):
            self._beta[ct][tt][node][index_infinity_numerators[a]] = infinity_numerators[a] / sum_inf_numerators
            
        sum_diff = 0
        for action in self._set_of_actions[node]:
            pi_prev = self._pi[ct][tt][node][action]
            self._pi[ct][tt][node][action] += self._mu[ct][tt][node] * (self._beta[ct][tt][node][action] - pi_prev)
            sum_diff += abs(self._pi[ct][tt][node][action] - pi_prev)
        sum_diff /= len(self._set_of_actions[node])
        
        self._time_slot_per_node[ct][tt][node] += 1
        
        
        # Choose strategy based on the probability distribution self._pi
        chosen_strategy = np.random.choice(list(range(len(self._set_of_actions[node]))), 1, p=list(self._pi[ct][tt][node].values()))
        sf = self._set_of_actions[node][int(chosen_strategy)][0]
        tp = self._set_of_actions[node][int(chosen_strategy)][1]
        self.parent._nodes[node].set_SF(sf)
        self.parent._nodes[node].set_TP(tp)
        self._current_action[node] = (sf, tp)
        
        # Check convergence with delta = 0.01
        if sum_diff <= 0.01:
            return True
        else:
            return False