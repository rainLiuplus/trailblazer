#!/usr/bin/env python
# Author: Ruediger Birkner (Networked Systems Group at ETH Zurich)

import logging
import numpy as np
from numpy.core.defchararray import index
import pandas as pd
from ipaddress import IPv4Network
import os
import random
import time
import subprocess
import signal
import socket


from config2spec.backend.minesweeper import MinesweeperBackend
from config2spec.dataplane.batfish_engine import BatfishEngine
from config2spec.netenv.network_environment import NetworkEnvironment
from config2spec.netenv.enumerate_sampler import EnumerateSampler
from config2spec.netenv.link_set_sampler import BlockLinksPolicySetSampler
from config2spec.netenv.merge_set_sampler import MergePolicySetSampler
from config2spec.netenv.random_set_sampler import RandomPolicySetSampler
from config2spec.netenv.random_sampler import RandomSampler
from config2spec.netenv.set_sampler import PolicySetSampler
from config2spec.netenv.sum_sampler import PolicySumSampler
from config2spec.policies.policy_db import PolicyDB
from config2spec.policies.policy_db import PolicyStatus
from config2spec.topology.builder.minesweeper_builder import BackendTopologyBuilder
from config2spec.topology.links import Link
from config2spec.topology.links import LinkState


def get_logger(name, loglevel):
    # LOGGING
    if loglevel == "INFO":
        log_level = logging.INFO
    elif loglevel == "DEBUG":
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    # add handler
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    if len(logger.handlers) == 0:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def init_manager(backend_path, port):
    ms_cmd = ['java', '-cp', '%s' % backend_path, 'org.batfish.backend.Backend', '%d' % port]

    # creating minesweeper manager
    ms_manager = MinesweeperManager(ms_cmd, port)
    ms_manager.start()

    return ms_manager


def init_backend(ms_manager, scenario, base_path, config_path, port):
    # init the backend (Batfish/Minesweeper)
    ms_backend = MinesweeperBackend(base_path, scenario, config_path, url="http://localhost", port=port, debug=False)
    ms_backend.init_minesweeper()

    ms_manager.backend = ms_backend

    ms_manager.restart(backend_calls=0, force=True)


def build_network(backend, scenario_path, max_failures, waypoints_min, waypoints_fraction, fix_seed=False):
    topology_files = backend.get_topology()
    network = BackendTopologyBuilder.build_topology(topology_files, scenario_path)

    # get waypoints
    all_routers = sorted(network.nodes())
    num_waypoints = max(waypoints_min, int(len(all_routers) / waypoints_fraction))  
    if fix_seed:
        random.seed(0)
        waypoints = random.sample(all_routers, num_waypoints)
    else:
        waypoints = random.sample(all_routers, num_waypoints)

    links = list()
    all_edges = network.get_undirected_edges()
    for i, edge in enumerate(all_edges):
        links.append(Link("l{id}".format(id=i), edge, LinkState.SYMBOLIC))

    # create the network environment
    netenv = NetworkEnvironment(links, k_failures=max_failures)

    return network, netenv, waypoints


def init_dp_engine(network, fib_path, debug=False):
    nodes = list(network.nodes())
    next_hops = network.next_hops
    simple_acls = network.simple_acls
    dp_engine = BatfishEngine(nodes, next_hops, simple_acls, fib_path, debug=debug)

    return dp_engine


def get_sampler(sampling_mode, netenv, policy_db, seed):
    if sampling_mode == "random":
        sampler = RandomSampler(netenv, seed=seed,
                                use_provided_samples=False, debug=False)
    elif sampling_mode == "sum":
        sampler = PolicySumSampler(netenv, policy_db, seed=seed,
                                   use_provided_samples=False, debug=False)
    elif sampling_mode == "enumerate":
        sampler = EnumerateSampler(netenv, seed=seed,
                                   use_provided_samples=False, debug=False)
    elif sampling_mode == "merge":
        sampler = MergePolicySetSampler(netenv, policy_db, seed=seed,
                                        use_provided_samples=False, debug=False)
    elif sampling_mode == "block":
        sampler = BlockLinksPolicySetSampler(netenv, policy_db, seed=seed,
                                             use_provided_samples=False, debug=False)
    elif sampling_mode == "randomset":
        sampler = RandomPolicySetSampler(netenv, policy_db, seed=seed,
                                         use_provided_samples=False, debug=False)
    else:
        sampler = PolicySetSampler(netenv, policy_db, seed=seed,
                                   use_provided_samples=False, debug=False)
    return sampler


def get_policy_db(network, waypoints=None, spec_pols=None, recursive_depth=-1, debug=False):
    return PolicyDB(network, waypoints=waypoints, spec_pols=spec_pols, recursive_depth=recursive_depth, debug=debug)


def get_init_spec(spec_path):
    spec_file = os.path.join(os.sep.join(spec_path.split(os.sep)[:-1]), 'policies.csv')
    if not os.path.exists(spec_file):
        return None
    spec_policies = pd.read_csv(spec_file, ',', index_col=[0, 1, 2, 3])
    return spec_policies


def recounstruct_destination(dest_str):
    from config2spec.policies.policy import PolicyDestination
    dest = dest_str.strip().replace("{", '').replace('}', '')
    router = dest.split(':')[0].strip()
    interface = dest.split(':')[1].split('(')[0].strip()
    subnet = IPv4Network(dest.split('(')[1].split(')')[0].strip())
    return PolicyDestination(router, interface, subnet)


def randomize_spec(spec_path, random_prop=0.05):
    from collections import defaultdict
    from config2spec.policies.policy import PolicySource, PolicyType

    spec_pols = get_init_spec(spec_path)
    if spec_pols is None:
        return None

    unique_policies = defaultdict(set)
    status_list = list()
    for idx in spec_pols.index:
        if idx[0].endswith('Reachability'):
            type = PolicyType.Reachability
        elif idx[0].endswith('Isolation'):
            type = PolicyType.Isolation
        elif idx[0].endswith("Waypoint"):
            type = PolicyType.Waypoint
        elif idx[0].endswith("LoadBalancingSimple"):
            type = PolicyType.LoadBalancingSimple
        elif idx[0].endswith("LoadBalancingEdgeDisjoint"):
            type = PolicyType.LoadBalancingEdgeDisjoint
        elif idx[0].endswith("LoadBalancingNodeDisjoint"):
            type = PolicyType.LoadBalancingNodeDisjoint

        species = int(idx[2]) if idx[2].isnumeric() else idx[2]
        instanced_idx = (type, IPv4Network(idx[1]), species, PolicySource(idx[3]))

        if spec_pols.loc[idx]['Status'].endswith('HOLDS'):
            status_list.append(PolicyStatus.HOLDS)
        elif spec_pols.loc[idx]['Status'].endswith('HOLDSNOT'):
            status_list.append(PolicyStatus.HOLDSNOT)
        else:
            status_list.append(PolicyStatus.UNKNOWN)

        destinations = spec_pols.loc[idx]['Destinations']
        for dest in destinations.split(','):
            inst_dst = recounstruct_destination(dest)
            unique_policies[instanced_idx].add(inst_dst)

    unique_index = list(unique_policies.keys())
    policy_index = pd.MultiIndex.from_tuples(unique_index, names=["type", "subnet", "specifics", "source"])

    envs = [{spec_pols['Environments'][0].split('{')[1].split('}')[0].strip()}] * len(unique_policies)
    destinations = list(unique_policies.values())
    new_df = pd.DataFrame({
        "Status": pd.Series(status_list, index=policy_index),
        "Environments": pd.Series(envs, index=policy_index),
        "Destinations": pd.Series(destinations, index=policy_index),
        "Original Status": pd.Series(status_list, index=policy_index),
    })

    new_df['Sources'] = new_df.index.get_level_values("source")
    rand_pols = new_df

    hold_num = len(rand_pols[rand_pols['Status']==PolicyStatus.HOLDS])
    rand_ides = np.random.randint(hold_num, size=int(hold_num * random_prop))
    hold_pols = rand_pols[rand_pols['Status']==PolicyStatus.HOLDS]
    unhold_pols = rand_pols[rand_pols['Status']==PolicyStatus.HOLDSNOT]
    for idx in rand_ides:
        hold_pols.at[hold_pols.index[idx], 'Status'] = PolicyStatus.HOLDSNOT
        unhold_pols.at[unhold_pols.index[idx], 'Status'] = PolicyStatus.HOLDS

    rand_pols = pd.concat([hold_pols, unhold_pols])
    rand_pols = rand_pols[rand_pols['Status']==PolicyStatus.HOLDS]
    rand_pols['Status'] = PolicyStatus.UNKNOWN
    rand_pols.sort_index()
    rand_pols.to_csv(os.path.join(spec_path, 'rand_policies.csv'), index=True)

    return rand_pols

class SlidingTimer(object):
    def __init__(self, window_size):
        self.window_size = window_size

        self.times = list()
        self.eliminated_policies = list()

        self.infinity = 10000000

    def mean_time_per_item(self):
        if self.times:
            mean_time = np.mean(self.times[-self.window_size:])
            return mean_time
        else:
            return 0

    def mean_time_per_policy(self):
        if self.times:
            total_time = sum(self.times[-self.window_size:])
            num_policies = sum(self.eliminated_policies[-self.window_size:])

            if num_policies > 0:
                return total_time/num_policies
            return self.infinity
        else:
            return 0

    def estimate_remaining_time(self, num_samples):
        mean_time = self.mean_time_per_item()

        if mean_time:
            return num_samples * mean_time
        else:
            return 0

    def update(self, runtime, num_eliminated_policies):
        self.times.append(runtime)
        self.eliminated_policies.append(num_eliminated_policies)

    def full_window(self):
        return len(self.times) == self.window_size


class Pipeline(object):
    def __init__(self, policy_db, sampler, dp_engine, netenv, ms_manager, window_size, network, pure_verify=False, max_failures=2, debug=False):
        self.logger = get_logger("", 'DEBUG' if debug else 'INFO')

        self.policy_db = policy_db
        self.sampler = sampler
        self.dp_engine = dp_engine
        self.netenv = netenv
        self.ms_manager = ms_manager

        self.pure_verify = pure_verify
        self.test_dsts = []

        self.forwarding_graphs = None

        self.network = network

        self.max_failures = max_failures

        # set up time analysis
        self.sampling_times = SlidingTimer(window_size)
        self.verification_times = SlidingTimer(window_size)
        
        # extra time analysis
        self.verification_correct_times = SlidingTimer(window_size)
        self.verification_wrong_times = SlidingTimer(window_size)

        # temporary variables
        self.prev_forwarding_graphs = None
        self.prev_guess_size = -1

    def sample(self, first=False):
        start_time = time.time()

        # pick the concrete env to use
        if first:
            concrete_env = self.sampler.get_all_up()
        elif self.sampler.fwd_state_based:
            concrete_env = self.sampler.get_next_env(self.prev_forwarding_graphs)
        else:
            concrete_env = self.sampler.get_next_env()

        # compute the dataplane and check the policies that hold
        num_eliminated_policies = -1

        if concrete_env:
            failed_edges = concrete_env.get_links(state=LinkState.DOWN)
            fib_file_name = self.ms_manager.get_dataplane(failed_edges)
            if fib_file_name:
                forwarding_graphs, the_path = self.dp_engine.get_forwarding_graphs(fib_file_name)
                if first:
                    self.policy_db.set_forwarding_graphs(forwarding_graphs)
                dominator_graphs = self.dp_engine.get_dominator_graphs()

                _, guess_size = self.policy_db.update_policies(9, forwarding_graphs, dominator_graphs)

                self.prev_forwarding_graphs = forwarding_graphs
                if self.prev_guess_size >= 0:
                    num_eliminated_policies = self.prev_guess_size - guess_size
                self.prev_guess_size = guess_size

            sampling_time = time.time() - start_time
            if num_eliminated_policies >= 0:
                self.sampling_times.update(sampling_time, num_eliminated_policies)

            return True
        else:
            self.logger.error("Couldn't find another unused sample!")
            return False


    def get_pruned_topos(self):
        self.sample(True)
        self.trim()
        num_topos, num_pruned_topos, _  = self.policy_db.get_pruned_topos(self.network, self.max_failures)
        return num_topos, num_pruned_topos


    def verify_query(self, query):
        start_time = time.time()
        num_policies = len(query.sources)

        # check policies with Minesweeper
        response = self.ms_manager.check_query(query)

        # update policy db
        if response.all_hold():
            verified_sources_dsts = response.holds()
            for source, dst in verified_sources_dsts:
                self.policy_db.update_policy(response.type, dst, response.specifics,
                                            PolicyStatus.HOLDS, source=source)
 
            verification_time = time.time() - start_time
            verified = len(response.holds())

            self.verification_correct_times.update(verification_time, verified)
        else:
            failed_sources_dsts = response.holds_not()
            for source, dst in failed_sources_dsts:
                self.policy_db.update_policy(response.type, dst, response.specifics,
                                            PolicyStatus.HOLDSNOT, source=source)

            if len(failed_sources_dsts)==0:
                print(query)


            verification_time = time.time() - start_time
            violated = len(response.holds_not())
            self.verification_wrong_times.update(verification_time,  violated)

        self.logger.info("Verify Time: {}".format(verification_time))

        # update timing stats
        verification_time = time.time() - start_time
        verified = len(response.holds())
        violated = len(response.holds_not())
        self.verification_times.update(verification_time, verified + violated)

        return True

    def verify(self):
        start_time = time.time()

        # pick next policy/policies to check
        if self.pure_verify:
            query = self.policy_db.get_query(environment=self.netenv, group=False)
        else:
            query = self.policy_db.get_query(environment=self.netenv, group=True)
        num_policies = len(query.sources)

        # check policies with Minesweeper
        response = self.ms_manager.check_query(query)

        # update policy db
        if response.all_hold():
            verified_sources_dsts = response.holds()
            for source, dst in verified_sources_dsts:
                self.policy_db.update_policy(response.type, dst, response.specifics,
                                            PolicyStatus.HOLDS, source=source)
            self.logger.info("Verify: Satisfied - All policies hold: {}".format(num_policies))     
            verification_time = time.time() - start_time
            verified = len(response.holds())
            self.logger.info("Verify Time: {}".format(verification_time))
            self.verification_correct_times.update(verification_time, verified)
        else:
            failed_sources_dsts = response.holds_not()
            for source, dst in failed_sources_dsts:
                self.policy_db.update_policy(response.type, dst, response.specifics,
                                            PolicyStatus.HOLDSNOT, source=source)

            self.logger.info("Verify: Counterexample - {} policies out of {} are violated".format(len(failed_sources_dsts),
                                                                                                   num_policies))
            verification_time = time.time() - start_time
            violated = len(response.holds_not())
            self.verification_wrong_times.update(verification_time,  violated)

        # update timing stats
        verification_time = time.time() - start_time
        verified = len(response.holds())
        violated = len(response.holds_not())
        self.verification_times.update(verification_time, verified + violated)

        return True

    def trim(self):
        connected_pairs = self.network.get_k_connected_routers(self.netenv.k_failures + 1)
        num_trimmed_policies = self.policy_db.trim_policies(connected_pairs)
        self.logger.debug("Trim: trimmed {} policies".format(num_trimmed_policies))

    def run(self, trim_policies=False, threshold=None):
        success = self.sample(first=True)


        if not success:
            self.logger.error("Dataplane sampling failed...")
            return

        # Trim policies first
        if trim_policies:
                self.logger.info("Trimming the policies.")
                trim_policies = False
                self.trim()

        # first check the original specification
        if self.policy_db.spec_pols is not None:
            pass
        elif self.pure_verify:
            pass
        else:
            self.logger.info("Running a couple of queries to init the verification timer.")
            # run enough queries to Minesweeper to match the window size of the timers
            while not self.verification_times.full_window():
                success = self.verify()

        self.logger.info("Starting the actual loop.")
        # start with elimination of dense violations using sampling, then decide whether to continue sampling or switch
        # to verification depending on the expected run time.

        dense = True
        sparse_sampling = True
        num_steps = 0
        verify_times = 0

        queries = self.policy_db.get_queries(self.netenv, 3000)

        if int(threshold) == -1:
            queries = queries = self.policy_db.get_queries(self.netenv, 3000)

        elif threshold is not None and int(threshold) < len(queries):
            random.seed(123456)
            queries = random.sample(queries, int(threshold))

        while True:
            remaining_samples = self.sampler.remaining_samples()
            remaining_policies = self.policy_db.num_policies(status=PolicyStatus.UNKNOWN)

            # for temp use
            if num_steps> len(queries) -1:
                break

            # logging
            if num_steps % 100 == 0:
                self.logger.info("Step {}: There are {} policies remaining.".format(num_steps, remaining_policies))
            num_steps += 1

            # check if we are done
            if remaining_samples == 0 or remaining_policies == 0:
                break

            # dense elimination - in the beginning we are in the dense elimination phase in which we try to quickly
            # narrow done the policy guess by using a few dataplane samples.
            sampling_time = self.sampling_times.mean_time_per_policy()
            verification_time = self.verification_times.mean_time_per_policy()

            if sampling_time <= verification_time:

                if not dense:
                    # self.logger.info("Switching back to dense elimination.")
                    dense = True

                if self.pure_verify:
                    success = self.verify_query(queries[num_steps-1])
                else:
                    success = self.sample()

            # sparse elimination - once the time needed to eliminate a policy by sampling is the same or less than
            # by verification, we switch to sparse elimination. In this mode, we decide based on an estimate of the
            # remaining time whether we want to continue sampling or start verifying.
            else:
                if dense:
                    dense = False

                total_sampling_time = self.sampling_times.estimate_remaining_time(remaining_samples)
                total_verification_time = self.verification_times.estimate_remaining_time(remaining_policies)

                # stick with sampling
                if total_sampling_time <= total_verification_time:
                    if not sparse_sampling:
                        sparse_sampling = True

                    if self.pure_verify:
                        success = self.verify_query(queries[num_steps-1])
                    else:
                        success = self.sample()

                # switch to verification
                else:
                    if sparse_sampling:
                        sparse_sampling = False

                    if trim_policies:
                        trim_policies = False
                        self.trim()
                    else:
                        verify_times += 1
                        success = self.verify_query(queries[num_steps-1])

            # check if something failed
            if not success:
                self.logger.error("Something failed (most likely the dataplane sampling)")
                break

        self.policy_db.policies = self.policy_db.policies[self.policy_db.policies['Status'] !=PolicyStatus.UNKNOWN]

        return


    def dump(self, path):
        with open(path, 'w') as f:
            for t in self.verification_times.times:
                f.write(str(t)+'\n')
        return

class MinesweeperManager(object):
    def __init__(self, command, port):
        self.process = None
        self.command = command
        self.running = False
        self.logger = get_logger('MinesweeperManager', 'INFO')
        self.queries = 0
        self.port = port

        self.backend = None

    def start(self):
        if not self.running:
            # self.logger.info("Starting Minesweeper.")

            # make sure the port is available before starting Minesweeper
            while MinesweeperManager.port_blocked(self.port):
                self.logger.info("Port %d still blocked." % (self.port, ))
                time.sleep(0.5)

            # start Minesweeper
            self.logger.info("Port %d finally free, everything ready to start Minesweeper." % (self.port, ))
            self.process = subprocess.Popen(self.command, preexec_fn=os.setsid)
            self.running = True

            # wait a bit for Minesweeper to fully start up
            time.sleep(2.0)
        else:
            self.logger.info("Minesweeper is already running.")

    def stop(self, backend_calls, force_stop=False):
        self.queries += backend_calls
        if self.queries > 3500 or force_stop:
            self.logger.debug("Killing Minesweeper as it answered %d queries - (force stop %s)" % (self.queries,
                                                                                                  force_stop))
            os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
            self.running = False
            self.queries = 0
            return True
        else:
            self.logger.debug("Keeping Minesweeper running as it answered just %d queries so far." % (self.queries, ))
            return False

    @staticmethod
    def port_blocked(port):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        try:
            sock.bind(("0.0.0.0", port))
            result = False
        except:
            result = True

        sock.close()
        return result

    def restart(self, backend_calls=0, force=False):
        assert self.backend, "Cannot restart the backend without self.ms_backend being set."

        # restart the backend/process
        stopped = self.stop(backend_calls=backend_calls, force_stop=force)
        if stopped:
            self.start()

            # init the restarted backend
            self.backend.init_minesweeper(force_init=True)

            # request topology such that the backend has already parsed the config
            self.backend.get_topology()

    def get_dataplane(self, failed_edges):
        return self.backend.get_dataplane(failed_edges)

    def check_query(self, query):
        # check if we should restart the backend
        self.queries += 1

        if self.queries > 500:
            self.restart(force=True)

        return self.backend.check_query(query)