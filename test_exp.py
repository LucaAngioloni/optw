import pandas as pd
import argparse
from glob import glob
import os
import numpy as np
import csv

from otpw import window_duration, solve, user_plot, gantt_plot, verify_path

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
    if "_res" in file:
        continue
    df = pd.read_csv(file, sep=";")
    pos = df[[' x', ' y']].values / 1000.
    window_time = df[' arrival'].values

    time_scale = np.max(window_time) + window_duration + 1

    path, exec_time = solve(pos, window_time, alpha=args.alpha, beta=args.beta, time_scale=time_scale)
    is_ok = verify_path(pos, window_time, path, time_scale=time_scale, verbose=True)

    print(f"Path ok: {is_ok}")

    with open(file.replace(".txt", "") + "_res.txt", mode='w') as wf:
        writer = csv.writer(wf, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        writer.writerow(['id', 'x', 'y', 'arrival'])
        for p in path:
            writer.writerow([p[-1]+1, int(p[0][0]*1000), int(p[0][1]*1000), p[2]])

    users.append(len(path))
    times.append(exec_time)

    if show_plots:
        user_plot(pos)
        gantt_plot(window_time)
        user_plot(pos, path)


print(f"Avg n. of users served: {np.mean(users):.2f}")
print(f"Avg exec time: {np.mean(times):.3f} s")