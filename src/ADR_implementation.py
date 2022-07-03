#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 16:35:58 2021
@author: Verónica Toro-Betancur and Gopika Premsankar
"""
import numpy as np
import math

class ADR:
    
    def __init__(self, parent):
        self.parent = parent
        self.parent.configuration.configure_SF_allnodes(sfStrategy="minimum")
        self.parent.configuration.configure_TP_allnodes(tpStrategy="maximum")
        
        self._requiredSNR = {7: -7.5, 8: -10, 9: -12.5, 10: -15, 11: -17.5, 12: -20} # in dB
        self._background_noise = -123
        self.dBm2W = lambda dBm : 10**(dBm / 10.0) / 1000.0
                
    def ADR_NET(self, calculated_sigma, input_pathloss, node, force_ADR=False):
        if len(input_pathloss) >= 20 or force_ADR:
            sf = self.parent._nodes[node].get_SF()
            tp = self.parent._nodes[node].get_TP()
            
            pathloss = []
            interferers_pathloss = []
            for i in range(len(input_pathloss)):
                pathloss.append(self.dBm2W(input_pathloss[i][0]))
                aux = []
                for j in input_pathloss[i][1]:
                    aux.append(self.dBm2W(j))
                interferers_pathloss.append(sum(aux))
            input_SNR = []
            for i in range(len(pathloss)):
                noise = self.dBm2W(self._background_noise)
                SINR = pathloss[i] / (interferers_pathloss[i] + noise)
                input_SNR.append(10 * math.log10(SINR))
        
            SF_min = 7
            TP_min = 2
            TP_max = 14

            SNR_max = np.mean(input_SNR)
            SNR_req = self._requiredSNR[sf]
#            deviceMargin = 15
            deviceMargin = calculated_sigma + 8 # 8 was found empirically.
            SNR_margin = SNR_max - SNR_req - deviceMargin
            steps = np.floor(SNR_margin / 3)

            calculated_sf = sf
            calculated_tp = tp

            while steps > 0 and calculated_sf > SF_min:
                calculated_sf -= 1
                steps -= 1
            while steps > 0 and calculated_tp > TP_min:
                calculated_tp -= 3
                steps -= 1
            if calculated_tp < 2:
                calculated_tp = 2
            while steps < 0 and calculated_tp < TP_max:
                calculated_tp += 3
                steps += 1
            if calculated_tp > 14:
                calculated_tp = 14

            if calculated_tp != tp or calculated_sf != sf or force_ADR:
                # Send ADR command only if ADR is forced or new parameters are calculated
                self.parent._nodes[node].set_SF(calculated_sf)
                self.parent._nodes[node].set_TP(calculated_tp)
                return 1
            else:
                return 0