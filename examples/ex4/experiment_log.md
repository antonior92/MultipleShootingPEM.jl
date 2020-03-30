# Experiment Batch #1

## Experiment 1 (11:23 06/03/2020) - DONE! 
In order to compare all experiments for similar configurations
- Last Commit: "Change config" (``d7c50483d8a982633455300965c38c54785814dd``)
- Server: ``hydra1``
- Screen: ``multistage0.1``
- Conda env: ``pytorch``
- Command:
```bash
for std in "0.00" "0.01"; do for tt in interm fast slow; do for ss in 2 3 4 5 6 7 8 9 10 20; do python generate_solutions.py --n_process 16 --n_stage $ss --time_const $tt --type multistage --noise_std $std  --folder multistage; done; done; done
```
- Downloaded to: ``solutions/multistage``

## Experiment 2 (11:33 06/03/2020) - DONE!
In order to compare all experiments for similar configurations:
- Last Commit: "Change config" (``d7c50483d8a982633455300965c38c54785814dd``)
- Server: ``hydra3``
- Screen: ``multipleshoot_cp_10``
- Conda env: ``pytorch``
- Command:
```bash 
for std in "0.00" "0.01" "0.02" "0.03" "0.05" "0.1"; do for tt in interm fast slow; do for sl in 2 5 10 20; do python generate_solutions.py --n_process 16 --shoot_len $sl --noise_std $std --time_const $tt --type multipleshoot --initial_constr_penalty 10 --folder ./multipleshoot_cp_10; done; done; done
```
- Downloaded to: ``solutions/multipleshoot_cp_10``

## Experiment 3 (11:33 06/03/2020) - DONE
In order to compare all experiments for similar configurations:
- Last Commit: "Change config" (``d7c50483d8a982633455300965c38c54785814dd``)
- Server: ``titan1``
- Screen: ``multipleshoot_cp_100``
- Conda env: ``pytorch``
- Command:
```bash
for std in "0.03" "0.05" "0.1"; do for tt in interm fast slow; do for sl in 2 5 10 20; do python generate_solutions.py --n_process 16 --shoot_len $sl --noise_std $std --time_const $tt --type multipleshoot --initial_constr_penalty 100 --folder multipleshoot_cp_100; done; done; done
```
- Downloaded to: ``solutions/multipleshoot_cp_100``

## Experiment 4 (13:59 06/03/2020) - DONE
In order to compare all experiments for similar configurations:
- Last Commit: "Change config" (``d7c50483d8a982633455300965c38c54785814dd``)
- Server: ``titan2``
- Screen: ``noe``
- Conda env: ``pytorch``
- Command:
```bash
for std in "0.00" "0.01" "0.02" "0.03" "0.05" "0.1"; do for tt in interm fast slow; do python generate_solutions.py --n_process 16 --time_const $tt --noise_std $std  --type noe --folder noe; done; done
```
- Downloaded to: ``solutions/noe``


## Experiment 5 (14:05 06/03/2020) - DONE 
In order to compare all experiments for similar configurations:
- Last Commit: "Change config" (``d7c50483d8a982633455300965c38c54785814dd``)
- Server: ``titan2``
- Screen: ``narx``
- Conda env: ``pytorch``
- Command:
```bash
for std in "0.00" "0.01" "0.02" "0.03" "0.05" "0.1"; do for tt in interm fast slow; do python generate_solutions.py --n_process 16 --time_const $tt --noise_std $std  --type narx --folder narx; done; done
```
- Downloaded to: ``solutions/narx``

## \# Experiment 6 (13:11 09/03/2020) - DONE
In order to compare all experiments for similar configurations:
- Last Commit: "Change config" (``d7c50483d8a982633455300965c38c54785814dd``)
- Server: ``titan3``
- Screen: ``multipleshoot_cp_1``
- Conda env: ``pytorch``
- Command:
```bash
for std in "0.03" "0.05" "0.1"; do for tt in interm fast slow; do for sl in 2 5 10 20; do python generate_solutions.py --n_process 16 --shoot_len $sl --noise_std $std --time_const $tt --type multipleshoot --initial_constr_penalty 1 --folder multipleshoot_cp_1; done; done; done
```
- Downloaded to: ``solutions/multipleshoot_cp_1``

##  Experiment 7 (11:33 06/03/2020) - DONE
In order to compare all experiments for similar configurations:
- Last Commit: "Change config" (``d7c50483d8a982633455300965c38c54785814dd``)
- Server: ``titan2``
- Screen: ``multipleshoot_cp_0_1``
- Conda env: ``pytorch``
- Command:
```bash
for std in "0.03" "0.05" "0.1"; do for tt in interm fast slow; do for sl in 2 5 10 20; do python generate_solutions.py --n_process 16 --shoot_len $sl --noise_std $std --time_const $tt --type multipleshoot --initial_constr_penalty 0.1 --folder multipleshoot_cp_0.1; done; done; done
```
- Downloaded to: ``solutions/multipleshoot_cp_0.1``


## Experiment 8 (13:16 08/03/2020) - DONE!
Redundant with  experiment 1. Just in case experiment \#1 is interrupted
- Last Commit: "Change config" (``d7c50483d8a982633455300965c38c54785814dd``)
- Server: ``titan1``
- Screen: ``multistage0.3``
- Conda env: ``pytorch``
- Command:
```bash
for tt in interm fast slow; do for ss in 2 3 4 5 6 7 8 9 10 20; do python generate_solutions.py --n_process 16 --n_stage $ss --time_const $tt --type multistage --noise_std "0.03"  --folder multistage; done; done
```
- Downloaded to: ``solutions/multistage``

## Experiment 9 (13:27 08/03/2020) - DONE
Redundant with  experiment 1. Just in case experiment \#1 is interrupted
- Last Commit: "Change config" (``d7c50483d8a982633455300965c38c54785814dd``)
- Server: ``titan2``
- Screen: ``multistage0.2``
- Conda env: ``pytorch``
- Command:
```bash
for tt in interm fast slow; do for ss in 2 3 4 5 6 7 8 9 10 20; do python generate_solutions.py --n_process 16 --n_stage $ss --time_const $tt --type multistage --noise_std "0.02"  --folder multistage; done; done
```
- Downloaded to: ``solutions/multistage``

-------------
# Experiment Batch #2


## Experiment 1 (17:18 11/03/2020) - DONE!
Assess running time
- Server: ``titan1``
- Screen: ``multistage``
- Conda env: ``pytorch``
- Command:
```bash
for tt in interm fast slow; do for ss in 3 5 7 10 20; do python generate_solutions.py --n_process 16 --n_stage $ss --time_const $tt --type multistage --noise_std "0.05"  --folder multistage_fev; done; done
```
- Downloaded to: ``solutions/multistage_fev``


## Experiment 2 (17:18 11/03/2020) - unfinished
Assess running time
- Server: ``titan2``
- Screen: ``multistage``
- Conda env: ``pytorch``
- Command:
```bash
for tt in interm fast slow; do for ss in 40 80; do python generate_solutions.py --n_process 16 --n_stage $ss --time_const $tt --type multistage --noise_std "0.05"  --folder multistage_fev; done; done
```


##  Experiment 3 (17:18 11/03/2020) - DONE
Assess running time
In order to compare all experiments for similar configurations:
- Server: ``titan3``
- Screen: ``multipleshoot_cp_100``
- Conda env: ``pytorch``
- Command:
```bash
for tt in interm fast slow; do for sl in 2 5 10 20; do python generate_solutions.py --n_process 16 --shoot_len $sl --noise_std "0.05" --time_const $tt --type multipleshoot --initial_constr_penalty 100 --folder multipleshoot_cp_100_fev; done; done; done
```


##  Experiment 4 (17:18 11/03/2020) - Not needed!!
Assess running time. Repeat experiment 3
In order to compare all experiments for similar configurations:
- Server: ``hydra1``
- Screen: ``multipleshoot_cp_100``
- Conda env: ``pytorch``
- Command:
```bash
for tt in interm fast slow; do for sl in 2 5 10 20; do python generate_solutions.py --n_process 16 --shoot_len $sl --noise_std "0.05" --time_const $tt --type multipleshoot --initial_constr_penalty 100 --folder multipleshoot_cp_100_fev; done; done; done
```

## Experiment 5 (17:18 11/03/2020) - Not needed!!
Assess running time. Repeat experiment 3
- Server: ``hydra3``
- Screen: ``multistage``
- Conda env: ``pytorch``
- Command:
```bash
for tt in interm fast slow; do for ss in 3 5 7 10 20; do python generate_solutions.py --n_process 16 --n_stage $ss --time_const $tt --type multistage --noise_std "0.05"  --folder multistage_fev; done; done
```