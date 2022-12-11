#!/usr/bin/env python
# Author: Ruediger Birkner (Networked Systems Group at ETH Zurich)

import re
import sys
from collections import defaultdict
from ipaddress import IPv4Network, ip_address

from config2spec.netenv.network_environment import ConcreteEnvironment
from config2spec.policies.policy_db import PolicyStatus
from config2spec.topology.links import Link
from config2spec.utils.logger import get_logger


class Response(object):
    def __init__(self, query, ms_response, debug=False):
        self.logger = get_logger('Response', 'DEBUG' if debug else 'INFO')

        self.type = query.type
        self.sources = query.sources
        self.destinations = query.destination
        self.subnets = [dst.subnet for dst in query.destination]
        self.specifics = query.specifics
        self.netenv = query.environment

        self.counter_example = None
        self.result = defaultdict(defaultdict)

        # TODO: test
        self.full_link_paths = query.full_link_paths

        self.parse_response(ms_response)

    def __str__(self):
        if self.counter_example:
            output = "Counter Example: %s\n" % self.counter_example
            output += "Failed Ingresses: %s" % ', '.join([str(src) for i, src in enumerate(self.sources) if self.result[i] == PolicyStatus.HOLDSNOT])
        else:
            output = "Verified"

        return output

    def parse_response(self, response):
        router_to_source = dict()
        for source in self.sources:
            router_to_source[source.router] = source

        # extract counter example from output
        if response.startswith("Verified"):
            # everything holds, so also no counter-example

            # self.result = [PolicyStatus.HOLDS] * len(self.sources)
            for source in self.sources:
                for dst in self.destinations:
                    if self.result[source] is None:
                        self.result[source] = defaultdict()
                    self.result[source][dst.subnet] = PolicyStatus.HOLDS
            self.counter_example = None

        else:

            if response.startswith("Flow:"):
                failed_links, source_dests = self.parse_flow_counterexample(response)

            elif response.startswith("Counterexample"):
                failed_links, source_routers = self.parse_generic_counterexample(response)

            ## test
            elif response.startswith("Not held"):
                # only handle one source - one dest case TODO: N-N
                assert len(self.sources) ==1 and len(self.destinations) ==1
                for source in self.sources:
                    for dst in self.destinations:
                        if self.result[source] is None:
                            self.result[source] = defaultdict()
                        self.result[source][dst.subnet] = PolicyStatus.HOLDSNOT
                return

            else:
                self.logger.error("Unknown response: {response}".format(response=response))
                sys.exit(1)

            self.counter_example = ConcreteEnvironment.from_failed_links(self.netenv.links, failed_links)
            assert len(source_dests) > 0

            ingresses = set()
            for src_router in source_dests.keys():
                if src_router in router_to_source:
                    ingresses.add(router_to_source[src_router])
                else:
                    self.logger.error("We couldn't find an ingress or it didn't match any source...")

            for source in self.sources:
                for dst in self.destinations:
                    if self.result[source] is None:
                        self.result[source] = defaultdict()
                    self.result[source][dst.subnet] = PolicyStatus.UNKNOWN

            for src_router in source_dests.keys():
                for dst_subnet in source_dests[src_router]:
                    
                    for dst in self.destinations:
                        if ip_address(dst_subnet) in dst.subnet:
                            self.result[router_to_source[src_router]][dst.subnet] = PolicyStatus.HOLDSNOT 

                    # self.result[router_to_source[src_router]][dst_subnet] = PolicyStatus.HOLDSNOT                

            # for source in self.sources:
            #     self.result.append(PolicyStatus.HOLDSNOT if source in ingresses else PolicyStatus.UNKNOWN)

    @staticmethod
    def parse_flow_counterexample(message):
        assert message.startswith("Flow:")

        failed_links = set()
        # source_routers = set()
        source_dsts = defaultdict(list)

        first = True
        for item in re.split('\n\n', message):
            ingress = re.search('ingress:(.+?) vrf:', item)
            ip_addrs = re.findall("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", item) # TODO: subnet mask
            if ingress and ingress.group(1):
                # source_routers.add(ingress.group(1))
                
                if ip_addrs and ip_addrs[1]:
                    # TODO: only for test
                    ip = '.'.join(ip_addrs[1].split('.'))

                    # source_dsts[ingress.group(1)].append(IPv4Network(ip))
                    source_dsts[ingress.group(1)].append(ip)

                if first:
                    first = True
                    tmp_blacklist = re.search('edgeBlacklist=\[(.*?)\]', item)
                    if tmp_blacklist:
                        blacklist = tmp_blacklist.group(1)
                        edges = re.findall('<(.+?):(.+?),\s(.+?):(.+?)>', blacklist)

                        for router1, intf1, router2, intf2 in edges:
                            failed_links.add(Link.get_name(router1, router2))

        # return failed_links, source_routers
        return failed_links, source_dsts

    @staticmethod
    def parse_generic_counterexample(message):
        assert message.startswith("Counterexample")

        failed_links = set()
        source_routers = set()

        edges = re.findall('link\((.+?),(.+?)\)', message)
        for router1, router2 in edges:
            failed_links.add(Link.get_name(router1, router2))

        return failed_links, source_routers

    def all_hold(self):
        for src in self.result.keys():
            for status in self.result[src].values():
                if status != PolicyStatus.HOLDS:
                    return False
        return True
        # return all(status == PolicyStatus.HOLDS for status in self.result)

    def holds_not(self):
        sources_dests = list()
        for src in self.result.keys():
            for dst, status in self.result[src].items():
                if status == PolicyStatus.HOLDSNOT:
                    sources_dests.append((src, dst))
        return sources_dests
        # sources = list()
        # for i, source in enumerate(self.sources):
        #     if self.result[i] == PolicyStatus.HOLDSNOT:
        #         sources.append(source)
        # return sources

    def holds(self):
        sources_dests = list()
        for src in self.result.keys():
            for dst, status in self.result[src].items():
                if status == PolicyStatus.HOLDS:
                    sources_dests.append((src, dst))
        return sources_dests 
        # sources = list()
        # for i, source in enumerate(self.sources):
        #     if self.result[i] == PolicyStatus.HOLDS:
        #         sources.append(source)
        # return sources
