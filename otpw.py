import numpy as np
import matplotlib.pyplot as plt
from time import time

# Global vars -------------------------------------------------------
n_users = 100
time_scale = 4*60
start_coord = np.array([0.5, 0.5])
speed = 0.02
window_duration = 60.

alpha = 1. # heuristic coefficient
beta = 1. # kernel coefficient

# Global vars -------------------------------------------------------

def gantt_plot(window_time, window_duration=window_duration):
    x_gnt = [(t, window_duration) for t in window_time]
    y_gnt = [(i+0.5, 1) for i in range(len(window_time))]
    fig, gnt = plt.subplots(figsize=(14,9))
    gnt.set_xlabel('seconds')
    gnt.set_ylabel('User')
    gnt.grid(True)
    gnt.set_yticks(np.arange(len(window_time))+1)

    for x, y in zip(x_gnt, y_gnt):
        gnt.broken_barh([x], y)
    plt.show()

def user_plot(pos, path=None):
    plt.figure(1, figsize=(12,7))
    plt.scatter(pos[:,0], pos[:,1])
    if path is not None:
        from_pos = start_coord
        for p in path:
            p = p[0]
            plt.arrow(from_pos[0], from_pos[1], p[0]-from_pos[0], p[1]-from_pos[1], head_width=0.01, head_length=0.01, fc='r', ec='r')
            from_pos = p
    plt.show()

def get_heuristics(pos, window_time, window_duration=window_duration, speed=speed, beta=beta):
    h = []
    for i in range(len(window_time)):
        window_overlap = 1 - np.abs(np.minimum(window_time+window_duration, window_time[i]+window_duration) - np.maximum(window_time, window_time[i])) / window_duration
        time_dist = np.linalg.norm(pos - pos[i,:], axis=1) * window_overlap / speed
        h_i = np.exp(-beta*np.mean(time_dist))
        h.append(h_i)

    return np.array(h)

def solve(pos, window_time, alpha=alpha, beta=beta, time_scale=time_scale, window_duration=window_duration, speed=speed, start_coord=start_coord, verbose=False):
    remaining_time = time_scale + window_duration
    rem_pos = pos.copy()
    rem_win = window_time.copy()
    rem_ids = np.arange(pos.shape[0])
    now_pos = start_coord
    path = []
    total_elapsed = 0.

    start_time = time()
    while remaining_time > 0 and len(rem_pos) > 0:
        dist = np.linalg.norm(rem_pos - now_pos, axis=1) / speed
        w = np.array([(total_elapsed + dist[i]) > win and (total_elapsed + dist[i]) < (win + window_duration) for i, win in enumerate(rem_win)]) * 1
        if np.sum(w) == 0: # if no users available
            future = [(win, dist[i]) for i, win in enumerate(rem_win) if (total_elapsed + dist[i]) < win]
            if len(future) == 0:
                break
            min_id = np.argmin([f[0] for f in future])
            time_to_min = future[min_id][1]
            wait_time = future[min_id][0] - time_to_min - total_elapsed
            remaining_time = remaining_time - wait_time
            total_elapsed = total_elapsed + wait_time
            continue
        h = get_heuristics(rem_pos, rem_win, beta=beta)
        a_star = (np.exp(-beta*dist) + alpha*h) * w

        best = np.argmax(a_star)
        elapsed = dist[best]
        remaining_time = remaining_time - elapsed
        total_elapsed = total_elapsed + elapsed
        if total_elapsed >= time_scale + window_duration:
            break
        path.append((rem_pos[best], elapsed, total_elapsed, rem_ids[best]))

        now_pos = rem_pos[best]
        rem_pos = np.delete(rem_pos, best, axis=0)
        rem_win = np.delete(rem_win, best, axis=0)
        rem_ids = np.delete(rem_ids, best, axis=0)

    end_time = time()

    exec_time = end_time-start_time

    if verbose:
        print(f"execution time: {exec_time:.3f} s")
        print(f"users served: {len(path)}")

    return path, exec_time

def verify_path(pos, window_time, path, speed=speed, start_coord=start_coord, time_scale=time_scale, window_duration=window_duration, verbose=False):
    if path[-1][2] > time_scale + window_duration:
        if verbose:
            print("Exeeded time limit")
        return False

    for p in path:
        if p[2] < window_time[p[3]] or p[2] > window_time[p[3]] + window_duration:
            if verbose:
                print("Visited user outside of the time window")
            return False

    return True

if __name__ == '__main__':
    show_plots = True
    # Random initialization
    pos = np.random.random((n_users,2))
    window_time = np.random.random(n_users)*time_scale

    if show_plots:
        user_plot(pos)
        gantt_plot(window_time)

    path, exec_time = solve(pos, window_time, verbose=True)

    print(f"Path ok: {verify_path(pos, window_time, path, verbose=True)}")

    # print(path)

    if show_plots:
        user_plot(pos, path)
