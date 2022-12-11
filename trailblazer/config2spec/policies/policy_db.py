#!/usr/bin/env python
# Author: Ruediger Birkner (Networked Systems Group at ETH Zurich)

from collections import Counter
from collections import defaultdict
from enum import Enum
from re import T, subn
import networkx as nx
import pandas as pd
import os

from config2spec.backend.query import Query

from config2spec.policies.policy import Policy, PolicyDestination, PolicySource
from config2spec.policies.policy import PolicyType
from config2spec.policies.policy_guesser import PolicyGuesser

from config2spec.utils.logger import get_logger


class PolicyStatus(Enum):
    UNKNOWN = "unknown"
    HOLDS = "holds"
    HOLDSNOT = "holdsnot"


class PolicyDB(object):
    def __init__(self, network, waypoints=None, spec_pols=None,  recursive_depth=-1, debug=False):
        # initialize logging
        self.debug = debug
        self.logger = get_logger("PolicyDB", "DEBUG" if debug else "INFO")
        self.init = False

        self.policy_guesser = PolicyGuesser(network, waypoints=waypoints, debug=debug)
        self.keys = ["type", "subnet", "specifics", "source"]

        self.policies = None  # dataframe with the columns - type, src, dst, specifics, policy status, environments
        self.previous_size = -1  # stores the size of the current policy guess

        self.tmp_state = None
        
        # test TODO
        self.spec_pols = spec_pols
        self.graph_similarity = None
        self.forwarding_graphs = None
        self.recursive_depth = recursive_depth

    def set_forwarding_graphs_similarity(self, fg_sims):
        self.graph_similarity = fg_sims
    
    def set_forwarding_graphs(self, forwarding_graphs):
        self.forwarding_graphs = forwarding_graphs

    def update_policies(self, sample, forwarding_graphs, dominator_graphs, node_local_reachability=False):
        # get the policy guess
        policies = self.policy_guesser.get_policies(forwarding_graphs, dominator_graphs, node_local_reachability=node_local_reachability)
        change, previous_size = self.update_policies2(policies, sample)
        return change, previous_size

    def update_policies2(self, policies, sample):
        # create a dataframe from the policy guess
        # for this, we first need to make sure that we have unique keys in the dataframe
        unique_policies = defaultdict(set)
        for ptype, destination, specifics, source in policies:
            index = (ptype, destination.subnet, specifics, source)
            unique_policies[index].add(destination)

        unique_index = list(unique_policies.keys())
        policy_index = pd.MultiIndex.from_tuples(unique_index, names=self.keys)

        envs = [{sample}] * len(unique_policies)
        destinations = list(unique_policies.values())

        new_df = pd.DataFrame({
            "Status": pd.Series(PolicyStatus.UNKNOWN, index=policy_index),
            "Environments": pd.Series(envs, index=policy_index),
            "Destinations": pd.Series(destinations, index=policy_index),
        })

        # if this is the first policy guess, init the db
        if not self.init:
            self.init = True
            self.policies = new_df
            # self.prune_policies()
            change = 1.0
            self.previous_size = len(self.policies)
        # else merge the new guess with the old ones. The old one is considered to be the left and the new one the right
        # in the merge
        else:
            # all policies that overlap (exist in both)
            overlapping_indexes = policy_index.intersection(self.policies.index)
            for env in self.policies.loc[overlapping_indexes, "Environments"]:
                env.add(sample)

            destination_series = pd.Series([dests1.union(dests2) for dests1, dests2 in
                                            zip(self.policies.loc[overlapping_indexes, "Destinations"],
                                                new_df.loc[overlapping_indexes, "Destinations"])],
                                           index=overlapping_indexes)
            self.policies.loc[overlapping_indexes, "Destinations"] = destination_series

            # all policies which are already in the db, but don't hold for the given sample
            left_only_indexes = self.policies.index.difference(policy_index)
            self.policies.loc[left_only_indexes, "Status"] = PolicyStatus.HOLDSNOT

            # all policies which are not yet in the db, but hold for the given sample
            right_only_indexes = policy_index.difference(self.policies.index)
            envs = [{sample}] * len(right_only_indexes)
            policies = pd.DataFrame({
                "Status": pd.Series(PolicyStatus.HOLDSNOT, index=right_only_indexes),
                "Environments": pd.Series(envs, index=right_only_indexes),
                "Destinations": new_df.loc[right_only_indexes, "Destinations"]
            })
            self.policies = self.policies.append(policies)

            # computing the size of the policy guess
            current_size = len(self.policies[(self.policies["Status"] == PolicyStatus.HOLDS) |
                                             (self.policies["Status"] == PolicyStatus.UNKNOWN)])

            if self.previous_size == 0:
                change = 0.0
            else:
                change = float(self.previous_size - current_size)/float(self.previous_size)

            self.previous_size = current_size

        self.policies["Sources"] = self.policies.index.get_level_values("source")
        self.policies.sort_index()

        return change, self.previous_size

    def update_policy(self, policy_type, destination, specifics, status, both=False, ori_spec=False, source=None):

        if source:
            if both and self.spec_pols is not None:
                if not (policy_type, destination, specifics, source) in self.spec_pols.index:
                    print((policy_type, destination.compressed, specifics, source.router))
                self.spec_pols.at[(policy_type, destination, specifics, source), "Status"] = status
                if not (policy_type, destination, specifics, source) in self.policies.index:
                    print((policy_type, destination.compressed, specifics, source.router))
                self.policies.at[(policy_type, destination, specifics, source), "Status"] = status       
            elif ori_spec and self.spec_pols is not None:
                self.spec_pols.at[(policy_type, destination, specifics, source), "Status"] = status
            else:                
                self.policies.at[(policy_type, destination, specifics, source), "Status"] = status
        else:
            if both and self.spec_pols is not None:
                self.spec_pols.loc[(policy_type, destination, specifics), "Status"] = status
                self.policies.loc[(policy_type, destination, specifics), "Status"] = status
            elif ori_spec and self.spec_pols is not None:
                self.spec_pols.loc[(policy_type, destination, specifics), "Status"] = status
            else:
                self.policies.loc[(policy_type, destination, specifics), "Status"] = status

    def change_status(self, current_status, next_status):
        self.policies.loc[self.policies["Status"] == current_status, "Status"] = next_status

    def num_policies(self, status=None):
        if not self.init:
            return 0
        if status:
            num_policies = len(self.policies[self.policies["Status"] == status])
        else:
            num_policies = len(self.policies)
        return num_policies

    def get_spec_policies(self, environment, status=PolicyStatus.UNKNOWN):
        if not self.init or self.spec_pols is None:
            return None
        raw_policies = self.spec_pols[self.spec_pols["Status"] == status]
        raw_group = raw_policies.groupby(["type", "subnet", "specifics"], sort=False).aggregate({"Sources": list, "Destinations": list, "Status": list})

        query_set = []
        added_subnets = set()

        for index, row in raw_group.iterrows():
            policy_type, _, specifics = row.name
            destination = set.union(*row["Destinations"])
            sources = row["Sources"]
            dst_subnet = list(destination)[0].subnet
            policy_sources = sources
            policy_destinations = destination
            is_break = False
            for sim_subnet, score in self.graph_similarity[dst_subnet]:
                if sim_subnet in added_subnets and is_break:
                    # continue
                    break
                if (policy_type, sim_subnet, specifics) in raw_group.index:
                    if score > 0.5:
                        sim_policies = raw_policies.loc[(policy_type, sim_subnet, specifics)]
                        sim_sources = sim_policies["Sources"]
                        sim_destination = set.union(*sim_policies["Destinations"])
                        if len(set.intersection(set(policy_sources), set(sim_sources)))/len(policy_sources) < 0.9:
                            continue
                        is_break = True
                        policy_sources = set.intersection(set(policy_sources), set(sim_sources))
                        if dst_subnet in sim_sources:
                            policy_sources.add(dst_subnet)
                        if sim_destination in sources:
                            policy_sources.add(sim_destination)
                        
                        policy_destinations = policy_destinations|sim_destination

                        added_subnets = added_subnets|set([dst_subnet, sim_subnet])
                        
            if len(policy_destinations):
                query = Query(policy_type, sources, policy_destinations, specifics, environment, negate=False)
                query_set.append(query)
        return query_set


    def get_raw_policy_tmp(self, status=None, group=False, num=100):
        if not self.init:
            return None
        if status:
                raw_policies = self.policies[self.policies["Status"] == status]
        else:
                raw_policies = self.policies.iloc[0].name
        
        raw_policies = raw_policies.sort_values(by=['type', 'subnet', 'source', 'specifics'])

        policy_db = []
        idx = 0
        while (True):
            if idx >= len(raw_policies):
                break

            tmp_policy = raw_policies.iloc[idx].name
            destination = raw_policies.iloc[idx]["Destinations"]

            cur_fwd_graph = self.forwarding_graphs[tmp_policy[1]]
            for e in cur_fwd_graph.edges():
                cur_fwd_graph[e[0]][e[1]]['weight'] = 1
            fwd_path = nx.johnson(cur_fwd_graph, weight='weight')

            path_list = list()
            tmp_graph = nx.DiGraph()
            for source_node in [tmp_policy[3]]:
                # TODO: skip no sink situation (temperorily)
                if fwd_path[source_node.router].get('sink', None) is None:
                    continue
                for i, path in enumerate(nx.all_shortest_paths(cur_fwd_graph, source=source_node.router, target='sink')):
                    path_list.append('-'.join(path[:-1]))
                    tmp_graph.add_path(path[:-1])
            
            source_path_set = set(tmp_graph.edges())
            fwd_path_set = set(cur_fwd_graph.edges())

            full_link_paths = ';'.join(path_list)


            raw_policy = (tmp_policy[0], [tmp_policy[3]], destination, tmp_policy[2], full_link_paths)

            idx += 1
            if len(path_list) > 0:
                policy_db.append(raw_policy)
            

        return policy_db

    def get_raw_policy(self, status=None, ori_spec=False, group=False):
        if not self.init:
            return None
        if status:
            if ori_spec:
                if self.spec_pols is not None:
                    raw_policies = self.spec_pols[self.spec_pols["Status"] == status]
            else:
                raw_policies = self.policies[self.policies["Status"] == status]
        else:
            if ori_spec:
                if self.spec_pols is not None:
                    raw_policies = self.spec_pols.iloc[0].name
            else:
                raw_policies = self.policies.iloc[0].name

        if group:
            raw_group = raw_policies.groupby(["type", "subnet", "specifics"], sort=False).aggregate({"Sources": list, "Destinations": list, "Status": list}).iloc[0]
            raw_group.replace("", float("NaN"), inplace=True)
            raw_group.dropna(inplace=True)
            policy_type, subnet, specifics = raw_group.name
            destination = set.union(*raw_group["Destinations"])
            sources = raw_group["Sources"]
            
            cur_fwd_graph = self.forwarding_graphs[subnet]
            for e in cur_fwd_graph.edges():
                cur_fwd_graph[e[0]][e[1]]['weight'] = 1
            fwd_path = nx.johnson(cur_fwd_graph, weight='weight')

            path_list = list()
            tmp_graph = nx.DiGraph()
            for source_node in sources:

                if fwd_path[source_node.router].get('sink', None) is None:
                    continue
                
                path_list.append('-'.join(fwd_path[source_node.router]['sink'][:-1]))
                for i, n in enumerate(fwd_path[source_node.router]['sink'][:-1]):
                    tmp_graph.add_edge(n, fwd_path[source_node.router]['sink'][i+1])
            
            source_path_set = set(tmp_graph.edges())
            fwd_path_set = set(cur_fwd_graph.edges())
            ratio = len(source_path_set)/len(fwd_path_set)

            if ratio <= 0.3:
                full_link_paths = ';'.join(path_list)
            else:
                full_link_paths = ''

            # last is the string of full link paths
            raw_policy = (policy_type, sources, destination, specifics, full_link_paths)
        else:
            tmp_policy = raw_policies.iloc[0].name

            destination = raw_policies.iloc[0]["Destinations"]

            cur_fwd_graph = self.forwarding_graphs[tmp_policy[1]]
            for e in cur_fwd_graph.edges():
                cur_fwd_graph[e[0]][e[1]]['weight'] = 1
            fwd_path = nx.johnson(cur_fwd_graph, weight='weight')

            path_list = list()
            tmp_graph = nx.DiGraph()
            for source_node in [tmp_policy[3]]:
                # TODO: skip no sink situation (temperorily)
                if fwd_path[source_node.router].get('sink', None) is None:
                    continue
                path_list.append('-'.join(fwd_path[source_node.router]['sink'][:-1]))
                for i, n in enumerate(fwd_path[source_node.router]['sink'][:-1]):
                    tmp_graph.add_edge(n, fwd_path[source_node.router]['sink'][i+1])
            
            source_path_set = set(tmp_graph.edges())
            fwd_path_set = set(cur_fwd_graph.edges())
            ratio = len(source_path_set)/len(fwd_path_set)
            
            full_link_paths = ';'.join(path_list)

            raw_policy = (tmp_policy[0], [tmp_policy[3]], destination, tmp_policy[2], full_link_paths)

        return raw_policy

    def get_policy(self, status=None, group=False):
        raw_policy = self.get_raw_policy(status=status, group=group)
        policy = Policy.get_policy(raw_policy[0], raw_policy[1], [raw_policy[2]], raw_policy[3])
        return policy

    def get_all_policies(self, status=None):
        if not self.init:
            return list()
        if status:
            policies = self.policies[self.policies["Status"] == status]
        else:
            policies = self.policies

        if not policies.empty:
            # policy_type, sources, destinations, specifics
            return policies.apply(lambda row: Policy.get_policy(row.name[0], [row.name[3]], list(row["Destinations"]), row.name[2]), axis="columns").tolist()
        else:
            return list()

    def get_query(self, environment, ori_spec=False, group=False):
        policy_type, sources, destinations, specifics, full_link_paths = self.get_raw_policy(status=PolicyStatus.UNKNOWN, ori_spec=ori_spec, group=group)
        query = Query(policy_type, sources, destinations, specifics, environment, full_link_paths, negate=False)
        return query
    
    def get_queries(self, environment,  num, group=False):
        pol_db = self.get_raw_policy_tmp(status=PolicyStatus.UNKNOWN, group=group, num=num)
        queries = []
        for pol in pol_db:
            policy_type, sources, destinations, specifics, full_link_paths = pol
            # cancel full link paths if recursive depth is -1
            full_link_paths = ''  if self.recursive_depth == -1 else full_link_paths
            query = Query(policy_type, sources, destinations, specifics, environment, full_link_paths, depth=self.recursive_depth, negate=False)
            queries.append(query)
        return queries


    def get_pruned_topos(self, network, max_failures, group=False):
        from scipy.special import comb
        if not self.init:
            return None

        raw_policies = self.policies[self.policies["Status"] == PolicyStatus.UNKNOWN]

        if group:
            pass
        else:
            num_pruned_topos_total = 0
            pol_pruned_topo_dict = dict()
            num_topos_total = 0
            for idx, tmp_policy in raw_policies.iterrows():
                num_pruned_topos = 0
                num_topos = 0
                tmp_policy = raw_policies.loc[idx].name

                cur_fwd_graph = self.forwarding_graphs[tmp_policy[1]]
                for e in cur_fwd_graph.edges():
                    cur_fwd_graph[e[0]][e[1]]['weight'] = 1
                fwd_path = nx.johnson(cur_fwd_graph, weight='weight')

                path_list = list()
                tmp_graph = nx.DiGraph()
                for source_node in [tmp_policy[3]]:
                    # TODO: skip no sink situation (temperorily)
                    if fwd_path[source_node.router].get('sink', None) is None:
                        continue
                    path_list.append('-'.join(fwd_path[source_node.router]['sink'][:-1]))
                    for i, n in enumerate(fwd_path[source_node.router]['sink'][:-1]):
                        tmp_graph.add_edge(n, fwd_path[source_node.router]['sink'][i+1])
                
                source_path_set = set(tmp_graph.edges())

                all_path_set_size = len(network.get_links())
                ratio = len(source_path_set)/all_path_set_size

                for k in range(1, max_failures+1):
                    num_pruned_topos += comb(all_path_set_size-len(source_path_set), k)
                    num_topos += comb(all_path_set_size, k)

                # pol_pruned_topo_dict[raw_policy] = num_pruned_topos
                num_pruned_topos_total += num_pruned_topos
                num_topos_total += num_topos

        return num_topos_total, num_pruned_topos_total, pol_pruned_topo_dict

    def get_source_counts(self, status=None):
        if not self.init:
            return None
        if status:
            raw_policies = self.policies[self.policies["Status"] == status]
        else:
            raw_policies = self.policies.iloc[0].name

        raw_group = raw_policies.groupby(["subnet"], sort=False).aggregate({"Sources": list})

        counts = dict()
        for index, row in raw_group.iterrows():
            sources = row.values[0]
            counts[index] = Counter(sources)

        return counts

    def trim_policies(self, k_connected_pairs):
        policy_count = 0

        policies = self.policies[self.policies["Status"] == PolicyStatus.UNKNOWN]
        for row in policies.iterrows():
            index = row[0]

            policy_type = index[0]
            src_router = index[3].router
            dst_subnet = index[1]
            specifics = index[2]

            destinations = row[1]["Destinations"]
            dst_routers = [destination.router for destination in destinations]

            if policy_type != PolicyType.Isolation:
                assert len(dst_routers) == 1, "There is more than one router connected to this subnet"
                dst_router = dst_routers[0]

                if src_router < dst_router:
                    pair = (src_router, dst_router)
                else:
                    pair = (dst_router, src_router)

                if pair not in k_connected_pairs:
                    self.policies.at[index, "Status"] = PolicyStatus.HOLDSNOT
                    policy_count += 1
        # test
        self.policies = self.policies[self.policies['Status'] !=PolicyStatus.HOLDSNOT]
        return policy_count

    def create_checkpoint(self):
        self.tmp_state = self.policies["Status"].copy()

    def restore_checkpoint(self):
        self.policies["Status"] = self.tmp_state

    def dump(self, file_path):
        if self.spec_pols is None:
            self.policies['Destinations'].apply(lambda x: str(x))
            self.policies.to_csv(file_path, index=True)
        else:
            self.policies['Destinations'].apply(lambda x: str(x))
            self.policies.to_csv(os.sep.join(file_path.split(os.sep)[:-1]) + os.sep + 'mined_spec.csv', index=True)
            self.spec_pols['Destinations'].apply(lambda x: str(x))
            self.spec_pols.to_csv(os.sep.join(file_path.split(os.sep)[:-1]) + os.sep + 'verified_spec.csv', index=True)


    def prune_policies(self):
        '''
        Prune the initial policies by the original specification 
        '''

        if self.spec_pols is None:
            return
        print("origin %d" %self.num_policies(PolicyStatus.UNKNOWN)) 
        print("spec size", len(self.spec_pols['Status']==PolicyStatus.HOLDS))

        combined = self.policies.append(self.spec_pols)
        self.policies = combined[~combined.index.duplicated(keep=False)]
        
        print("remained %d" %self.num_policies(PolicyStatus.UNKNOWN))


    def update_spec_pols(self):
        import time
        start = time.time()

        if self.spec_pols is None:
            return
                
        proved_policies = self.policies[self.policies['Status']!=PolicyStatus.UNKNOWN]
        combined = self.spec_pols.append(proved_policies)
        self.spec_pols = self.spec_pols.drop(combined.index[combined.index.duplicated()].unique())

        duration = time.time() - start
        return duration