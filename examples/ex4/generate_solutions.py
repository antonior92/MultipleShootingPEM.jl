# %% Imports
from warnings import warn
import numpy as np
import tqdm
import argparse
from base import GenerateData
from narx import solve_osa
from noe import solve_frs
from multistage import solve_msa
from multipleshoot import solve_ms
from multiprocessing import Process, JoinableQueue, Pool
from os import path
import os
import time


parser = argparse.ArgumentParser()

parser.add_argument("--reps", type=int, default=100,
                    help='number of repeated experiments')
parser.add_argument("--noise_std", type=float, default=0.05,
                    help='output noise standard deviation')
parser.add_argument("--type", default='narx',
                    help='predictor being analysed. Options are:'
                         '{narx, noe, multistage, multipleshoot}.')
parser.add_argument("--time_const", default='interm',
                    help="Set the time constant of the system being estimated. Options are:"
                         "{interm, fast, slow}.")
parser.add_argument("--n_stage",  type=int, default=10,
                    help='number of stages used in multistage prediction.')
parser.add_argument("--shoot_len",  type=int, default=2,
                    help='maximum simulation length used for multiple shooting estimation.')
parser.add_argument("--n_process",  type=int, default=4,
                    help='number of parallel processes')
parser.add_argument("--append",  action='store_true',
                    help='append solution to exist files if possible')
parser.add_argument("--seed", type=int, default=0,
                    help='seed to use in the first experiments will use an increment on that.')
parser.add_argument('--folder', default='./solutions',
                    help='output folder (default: ./solutions)')
parser.add_argument('--initial_constr_penalty', default=1.0, type=float,
                    help="initial penalty constraint for multiple shooting estimation. "
                         "See scipy.optimize.minimize(method=’trust-constr’) documentation")
parser.add_argument('--initial_trust_radius', default=1.0, type=float,
                    help="initial penalty constraint for multiple shooting estimation. "
                         "See scipy.optimize.minimize(method=’trust-constr’) documentation")
args, unk = parser.parse_known_args()
print(args)
# Check for unknown options
if unk:
    warn("Unknown arguments:" + str(unk) + ".")

# %% Saver
# Define file name
name = args.type
if args.type == 'multistage':
    name += '_' + str(args.n_stage)
if args.type == 'multipleshoot':
    name += '_' + str(args.shoot_len)
name += '_' + str(args.noise_std)
name += '_' + str(args.time_const)
name += '_solutions.csv'
name = os.path.join(args.folder, name)
# Check folder
if not path.exists(args.folder):
    os.makedirs(args.folder)
# write header
if not (args.append and path.isfile(name)):
    with open(name, 'w+') as out:
        out.write('y[k-1],y[k-2],u[k-1],exec time,seed\n')


def saver(q):
    while True:
        val = q.get()
        if val is None:
            break
        sol, total_time, seed = val
        pbar.update(1)
        with open(name, 'a') as out:
            out.write(','.join([str(x) for x in sol]) + ',' + str(total_time) + ',' + str(seed) + '\n')
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
    elif args.type == 'multipleshoot':
        sol = solve_ms(u, y, N, ny, nu, args.shoot_len, theta0,
                       initial_constr_penalty=args.initial_constr_penalty,
                       initial_trust_radius=args.initial_trust_radius)
    total_time = time.time() - start
    q.put([sol, total_time, seed])


pbar = tqdm.tqdm(initial=0, total=args.reps, smoothing=0.05)
q = JoinableQueue()
p = Process(target=saver, args=(q,))
p.start()
pool = Pool(processes=args.n_process)
pool.map(solve, range(args.seed, args.reps))
q.put(None)  # Poison pill
q.join()
p.join()