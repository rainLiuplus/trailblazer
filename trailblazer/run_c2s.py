#!/usr/bin/env python
# Author: Ruediger Birkner (Networked Systems Group at ETH Zurich)

import argparse
import os
import time

from helper import build_network
from helper import get_logger
from helper import get_policy_db
from helper import get_sampler
from helper import init_backend
from helper import init_dp_engine
from helper import init_manager
from helper import randomize_spec
from helper import Pipeline

''' main '''
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario_path', help='path to scenario', type=str)
    parser.add_argument('--backend_path', default=os.path.sep.join(['batfish', 'projects', 'backend', 'target/backend-bundle-0.36.0.jar']), help='path to backend executable', type=str)
    parser.add_argument('--batfish_path', help='path to cloned Batfish GitHub repo', default='tmp', type=str)
    parser.add_argument('-p', '--port', help='port batfish is listening on', type=int, default=8192)
    parser.add_argument('-d', '--debug', help='enable debug output', action='store_true')
    parser.add_argument('-r', '--randomize', help='randomize the correct specification to simulate the given incorrect spec and mine policies on it.', action='store_true')
    parser.add_argument('-pv', '--pure_verify', help='Mine policies only by minesweeper', action='store_true', default=True)

    parser.add_argument('-mf', '--max_failures', help='maximum amount of failures to allow', type=int)
    parser.add_argument('-rd', '--recursive_depth', help='depth for recursive best path optimization')
    parser.add_argument('-t', '--threshold', help='number of tested policies', default=1, type=int)

    args = parser.parse_args()

    # init logger
    debug = args.debug
    logger = get_logger("Config2Spec", 'DEBUG' if debug else 'INFO')

    # name of the scenario
    scenario = os.path.basename(args.scenario_path)

    # all the necessary paths where config, fib and topology files are being stored
    batfish_path = args.batfish_path  # path to cloned Batfish repo directory
    backend_path = args.backend_path  # path to Batfish executable
    scenario_path = args.scenario_path
    batfish_port = args.port

    recursive_depth = int(args.recursive_depth)

    start = time.time()
    # create backend manager
    ms_manager = init_manager(backend_path, batfish_port)

    # general settings
    seed = 8006
    window_size = 10
    sampling_mode = "sum"
    trimming = True
    waypoints_min = 3
    waypoints_fraction = 5

    # maximum amount of failures to be allowed
    if args.max_failures is None:
        max_failures = 1
    else:
        max_failures = args.max_failures

    # all the necessary paths where config, fib and topology files are being stored
    config_path = os.path.join(scenario_path, "configs")
    fib_path = os.path.join(scenario_path, "fibs")
    spec_path = os.path.join(scenario_path, "spec")

    # initialize specification policies
    if args.randomize:
        spec_pols = randomize_spec(spec_path)
    else:   
        spec_pols = None

    # initialize the backend
    init_backend(ms_manager, scenario, batfish_path, config_path, batfish_port)

    # initialize all the data structures
    network, netenv, waypoints = build_network(ms_manager.backend, scenario_path, max_failures,
                                               waypoints_min, waypoints_fraction, fix_seed=True)
    dp_engine = init_dp_engine(network, fib_path, debug=args.debug)

    # init policy Database
    policy_db = get_policy_db(network, waypoints=waypoints, spec_pols=spec_pols, recursive_depth=recursive_depth, debug=debug)

    # get sampler
    sampler = get_sampler(sampling_mode, netenv, policy_db, seed)

    # run Config2Spec pipeline
    pipeline = Pipeline(policy_db, sampler, dp_engine, netenv, ms_manager, window_size, network, args.pure_verify, max_failures, debug)


    pipeline.run(trim_policies=trimming, threshold=args.threshold)

    # completely kill Minesweeper
    ms_manager.stop(0, force_stop=True)
    duration = time.time() - start
    print('duration: %f mins' % (duration/60))

    config_path = "k={}_d={}".format(max_failures, recursive_depth)
    
    cc_path = 'cc' if 'cc' in backend_path else 'wo-cc'

    dump_path = os.path.join(scenario_path, cc_path, config_path)
    os.makedirs(dump_path, exist_ok=True)

    policy_db.dump(os.path.join(dump_path, "policies.csv"))
    pipeline.dump(os.path.join(dump_path, "durations.txt"))

