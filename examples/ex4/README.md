# Example 4: **Comparing multiple shooting and multi-step-ahead prediciton**

This example compares multiple shooting and multi-step-ahead prediction. It also
include in the comparison standard NARX and NOE models.


## File structure

The following files are available in this folder

- `base.py`: contain the function used for data generation;
- `{multipleshoot, multistage, narx, noe}.py`: contain basic functions for parameter estimation, 
using, respectively, multiple shooting, multi-step-ahead prediction, standard NARX and NOE.
Just run `python {multipleshoot, multistage, narx, noe}.py` for one usage example of each case.
-`generate_solutions.py`: run parameter estimation several times for a given configuration,
 producing a `.csv` files containing one estimate per line.
-`plot_histogram.py`: use `.csv` file produced by `generate_solutions.py` to generate histogram 
for a given estimator.


## Running the experiment

In order to run the set of experiments described in the paper the following bash command sequence can be used:
```bash
for std in "0.00" "0.01" "0.02" "0.03"; do for tt in interm fast slow; do for sl in 2 5 10 20; do python generate_solutions.py --n_process 16 --shoot_len $sl --noise_std $std --time_const $tt --type multipleshoot --initial_trust_constr 100 --folder ./solutions_constr_penalty_100; done; done; done
for std in "0.00" "0.01" "0.02" "0.03"; do for tt in interm fast slow; do python generate_solutions.py --n_process 16 --time_const $tt --type narx; done; done
for std in "0.00" "0.01" "0.02" "0.03"; do for tt in interm fast slow; do python generate_solutions.py --n_process 16 --time_const $tt --type noe; done; done
for std in "0.00" "0.01" "0.02" "0.03"; do for tt in interm fast slow; do for ss in 2 3 4 5 6 7 8 9 10 20; do python generate_solutions.py --n_process 16 --n_stage $ss --time_const $tt --type multistage --folder multistage; done; done; done
```
