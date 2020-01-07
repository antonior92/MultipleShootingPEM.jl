# %% Imports
from warnings import warn
import numpy as np
import tqdm
import argparse
from base import GenerateData
from narx import solve_osa
from noe import solve_frs
from multistage import solve_msa
from multiprocessing import Process, JoinableQueue, Pool
from os import path
import time


parser = argparse.ArgumentParser()

parser.add_argument("--reps", type=int, default=100,
                    help='number of repeated experiments')
parser.add_argument("--noise_std", type=float, default=0.05,
                    help='output noise standard deviation')
parser.add_argument("--type", default='narx',
                    help='predictor being analysed. Options are:'
                         '{narx, noe, multistage}.')
parser.add_argument("--time_const", default='interm',
                    help="Set the time constant of the system being estimated. Options are:"
                         "{interm, fast, slow}.")
parser.add_argument("--n_stage",  type=int, default=10,
                    help='number of stages used in multistage prediction')
parser.add_argument("--n_process",  type=int, default=4,
                    help='number of parallel processes')
parser.add_argument("--append",  action='store_true',
                    help='append solution to exist files if possible')
parser.add_argument("--seed", type=int, default=0,
                    help='seed to use in the first experiments will use an increment on that.')
args, unk = parser.parse_known_args()
print(args)
# Check for unknown options
if unk:
    warn("Unknown arguments:" + str(unk) + ".")

# %% Saver
name = args.type
if args.type == 'multistage':
    name += '_' + str(args.n_stage)
name += '_' + str(args.noise_std)
name += '_' + str(args.time_const)
name += '_solutions.csv'

if not (args.append and path.isfile(name)):
    with open(name, 'w+') as out:
        out.write('y[k-1],y[k-2],u[k-1],exec time\n')

def saver(q):
    while True:
        val = q.get()
        if val is None:
            break
        sol, total_time = val
        pbar.update(1)
        with open(name, 'a') as out:
            out.write(','.join([str(x) for x in sol]) + ',' + str(total_time) + '\n')
        q.task_done()
    # Finish up
    q.task_done()


# %% Solution producer
gn = GenerateData(noise_std=args.noise_std, time_constant=args.time_const)
theta0 = [0.0, 0.0, 0.0]

def solve(seed):
    u, y, x0 = gn.generate(seed)
    N, ny, nu, = gn.N, gn.ny, gn.nu
    start = time.time()
    if args.type == 'narx':
        sol = solve_osa(u, y, x0, N, ny, nu, theta0)
    elif args.type == 'noe':
        sol = solve_frs(u, y, x0, N, ny, nu, theta0)
    elif args.type == 'multistage':
        sol = solve_msa(u, y, x0, N, ny, nu, args.n_stage, theta0)
    total_time = time.time() - start
    q.put([sol, total_time])


pbar = tqdm.tqdm(initial=0, total=args.reps, smoothing=0.05)
q = JoinableQueue()
p = Process(target=saver, args=(q,))
p.start()
pool = Pool(processes=args.n_process)
pool.map(solve, range(args.seed, args.reps))
q.put(None)  # Poison pill
q.join()
p.join()