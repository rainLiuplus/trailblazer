#!/usr/bin/env python
# Author: Ruediger Birkner (Networked Systems Group at ETH Zurich)

from numpy.core.numeric import full
from config2spec.policies.policy import PolicyType
# test
from config2spec.policies.policy import PolicySource


class Query(object):
    def __init__(self, query_type, sources, destination, specifics, environment, full_link_paths='', depth=1, negate=False):
        self.type = query_type
        self.str_type = ""
        self.sources = sources

        if isinstance(destination, set):
            # self.destination = list(destination)[0]
            self.destination = list(destination)
        else:
            self.destination = destination

        self.specifics = specifics

        self.environment = environment
        self.negate = negate

        self.full_link_paths = full_link_paths
        self.depth = depth

        self.attributes = dict()
        self.init()

    def init(self):
        if self.type == PolicyType.Reachability:
            self.str_type = "reachability"
            negate = False
        elif self.type == PolicyType.Isolation:
            self.str_type = "reachability"
            negate = True
        elif self.type == PolicyType.Waypoint:
            self.str_type = "waypoint"
            negate = False
        elif self.type == PolicyType.LoadBalancingSimple:
            self.str_type = "loadbalancing"
            negate = False
        elif self.type == PolicyType.LoadBalancingNodeDisjoint:
            self.str_type = "nodedisjointlb"
            negate = False
        else:
            return
        source_str = "|".join(["^{source}$".format(source=source) for source in self.sources])
        dest_router_str = "|".join(["{dst}".format(dst=dst.router) for dst in self.destination])
        # dest_inter_str = "|".join(["^{source}$".format(source=dst.interface) for dst in self.destination])
        self.attributes["IngressNodeRegex"] = source_str
        self.attributes["FinalNodeRegex"] = dest_router_str
        # self.attributes["FinalIfaceRegex"] = dest_inter_str
        # self.attributes["FinalNodeRegex"] = self.destination.router
        self.attributes["FinalIfaceRegex"] = self.destination[0].interface # TODO: multi interfaces
        self.attributes["Negate"] = self.negate != negate
        self.attributes["MaxFailures"] = self.environment.k_failures
        self.attributes["Environment"] = self.environment.get_polish_notation()

        # full link paths
        self.attributes["FullLinkPaths"] = self.full_link_paths

        # The prefix for computing new best path when a link fails
        self.attributes["Prefix"] = self.destination[-1].subnet
        # The max depth for recursively computing new best path
        self.attributes["Depth"] = self.depth

        # add policy specific features
        if self.type == PolicyType.Waypoint:
            if isinstance(self.specifics, list):
                self.attributes["Waypoints"] = ",".join(self.specifics)
            else:
                self.attributes["Waypoints"] = self.specifics
        elif self.type == PolicyType.LoadBalancingSimple or self.type == PolicyType.LoadBalancingNodeDisjoint:
            self.attributes["NumPaths"] = self.specifics

    def to_dict(self):
        output = self.attributes.copy()
        output["type"] = self.type

        return output

    def __str__(self):
        output = "{query_type} Query: \n".format(query_type=self.str_type, )
        for key, value in self.attributes.items():
            output += "\t{key}: {value}\n".format(key=key, value=value)
        return output

    def to_string_representation(self):
        output = "Type:{query_type};".format(query_type=self.str_type)
        for key, value in self.attributes.items():
            output += "{key}:{value};".format(key=key, value=value)
        return output
