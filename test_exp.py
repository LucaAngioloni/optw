import pandas as pd
import argparse
from glob import glob
import os
import numpy as np

from otpw import window_duration, solve, user_plot, gantt_plot

show_plots = False

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input",
    type=str,
    help="path to input folder")
ap.add_argument("-a", "--alpha",
    type=float,
    default=0.5,
    help="heuristic coefficient")
ap.add_argument("-b", "--beta",
    type=float,
    default=0.1,
    help="kernel coefficient")
args = ap.parse_args()

users = []
times = []
for file in glob(os.path.join(args.input, "*.txt")):
    df = pd.read_csv(file, sep=";")
    pos = df[[' x', ' y']].values / 1000.
    window_time = df[' arrival'].values

    time_scale = np.max(window_time) + window_duration + 1

    path, exec_time = solve(pos, window_time, alpha=args.alpha, beta=args.beta, time_scale=time_scale)

    users.append(len(path))
    times.append(exec_time)

    if show_plots:
        user_plot(pos)
        gantt_plot(window_time)
        user_plot(pos, path)


print(f"Avg n. of users served: {np.mean(users):.2f}")
print(f"Avg exec time: {np.mean(times):.3f} s")