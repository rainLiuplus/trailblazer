# Trailblazer

## Install

1. __Clone this repository__

2. __Build Batfish and Minesweeper__
Follow the steps described [here](batfish/README.md)

3. __Create a conda environment with all requirements__
    ```bash
    $ conda create -n yourenv python=3.6
    $ pip install -r requirements.txt
    ```


## Run
```
$ python run_c2s.py --scenario_path <scenario path> -mf <num of max dropped links> -rd <recursive depth> --threshold <num of policies to be verified>
```

#### Arguments

* __scenario path__ - The path to the directory containing the
scenario.

* __mf__ - An int specifying the maximum amount of failures to allow.

* __rd__ - An int specifying the depth for recursive best path optimization.

* __threshold__ - An int specifying the maximum number of policies to verify.

### Example

```bash
$ python run_c2s.py --scenario_path scenarios/bics/ospf -mf 2 -rd 1 --threshold 1
```
