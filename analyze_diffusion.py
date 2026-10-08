# -*- coding: utf-8 -*-
"""Simulasi model difusi pada jaringan dengan ndlib.

Mereproduksi temuan utama Module 4 Network Modeling and Analysis in Python
(University of Michigan): model Threshold, Independent Cascade, SIR, dan SIS
pada jaringan nyata, termasuk pengaruh pemilihan seed node terhadap penyebaran.
"""
import os
import operator
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
IMG = os.path.join(BASE, "charts")
os.makedirs(IMG, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.dpi"] = 120

G = nx.read_edgelist(os.path.join(DATA, "german.txt"), delimiter=" ")
print("german:", G.number_of_nodes(), "node,", G.number_of_edges(), "edge")


def seed_config(config, importance_measure, n):
    if importance_measure:
        s = sorted(importance_measure(G).items(), key=operator.itemgetter(1))[::-1]
        highest = [x for x, _ in s[:n]]
        config.add_model_initial_configuration("Infected", highest)
    else:
        config.add_model_parameter("fraction_infected", float(n) / len(G.nodes()))


def run_threshold(importance_measure=None, iterate=50, n=1, threshold=0.25):
    model = ep.ThresholdModel(G, seed=42)
    config = mc.Configuration()
    seed_config(config, importance_measure, n)
    for node in G.nodes:
        config.add_node_configuration("threshold", node, threshold)
    model.set_initial_status(config)
    return [it["node_count"][1] for it in model.iteration_bunch(iterate)]


def run_ic(importance_measure=None, iterate=50, n=1, threshold=0.25):
    model = ep.IndependentCascadesModel(G, seed=42)
    config = mc.Configuration()
    seed_config(config, importance_measure, n)
    for edge in G.edges():
        config.add_edge_configuration("threshold", edge, threshold)
    model.set_initial_status(config)
    iters = model.iteration_bunch(iterate)
    return [it["node_count"][1] + it["node_count"][2] for it in iters]


def run_sir(importance_measure=None, iterate=100, n=1, beta=0.1, gamma=0.05):
    model = ep.SIRModel(G, seed=42)
    config = mc.Configuration()
    seed_config(config, importance_measure, n)
    config.add_model_parameter("beta", beta)
    config.add_model_parameter("gamma", gamma)
    model.set_initial_status(config)
    iterations = model.iteration_bunch(iterate)
    total = [0] * iterate
    status = [0] * len(G.nodes())
    for item in iterations:
        for k, v in item["status"].items():
            if v == 1:
                status[int(k)] = 1
        total[item["iteration"]] = sum(status)
    return [it["node_count"][1] for it in iterations], total


def run_sis(iterate=100, n=1, beta=0.3, _lambda=0.2):
    model = ep.SISModel(G, seed=42)
    config = mc.Configuration()
    seed_config(config, None, n)
    config.add_model_parameter("beta", beta)
    config.add_model_parameter("lambda", _lambda)
    model.set_initial_status(config)
    return model.iteration_bunch(iterate)


METHODS = {
    "random": None,
    "degree": nx.degree_centrality,
    "closeness": nx.closeness_centrality,
    "betweenness": nx.betweenness_centrality,
    "pagerank": nx.pagerank,
}
COLORS = {"random": "#9CA3AF", "degree": "#1D4ED8", "closeness": "#0D9488",
          "betweenness": "#7C3AED", "pagerank": "#DC2626"}

# 1. Threshold model: 5 metode seed (I=50, N=5, T=0.3)
I, N, T = 50, 5, 0.3
curves = {m: run_threshold(fn, iterate=I, n=N, threshold=T) for m, fn in METHODS.items()}
final = {m: c[-1] for m, c in curves.items()}
print("threshold final:", final, "-> paling sedikit:", min(final, key=final.get))
fig, ax = plt.subplots(figsize=(8, 5))
for m, c in curves.items():
    ax.plot(c, label=m, color=COLORS[m], linewidth=2)
ax.set_xlabel("Iterasi"); ax.set_ylabel("Node terinfeksi")
ax.set_title("Model Threshold: pengaruh pemilihan seed node")
ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(IMG, "01_threshold_seeds.png")); plt.close(fig)

# 2. Threshold vs Independent Cascade (I=40, N=20, T=0.3, seed degree)
I2, N2, T2 = 40, 20, 0.3
n_thr = run_threshold(nx.degree_centrality, iterate=I2, n=N2, threshold=T2)[-1]
n_ic = run_ic(nx.degree_centrality, iterate=I2, n=N2, threshold=T2)[-1]
print(f"threshold={n_thr} IC={n_ic} selisih={n_thr - n_ic}")
fig, ax = plt.subplots(figsize=(6, 4.5))
ax.bar(["Threshold", "Independent\nCascade"], [n_thr, n_ic], color=["#1D4ED8", "#F59E0B"])
ax.set_ylabel("Node terinfeksi akhir")
ax.set_title("Threshold menyebar jauh lebih luas dari Independent Cascade")
for i, v in enumerate([n_thr, n_ic]):
    ax.text(i, v + 20, str(v), ha="center", fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(IMG, "02_threshold_vs_ic.png")); plt.close(fig)

# 3. SIR: terinfeksi saat ini dan total pernah terinfeksi (N=25, I=300)
N3, I3, b3, g3 = 25, 300, 0.1, 0.05
sir_now, sir_tot = {}, {}
for m, fn in METHODS.items():
    now, tot = run_sir(fn, iterate=I3, n=N3, beta=b3, gamma=g3)
    sir_now[m], sir_tot[m] = now, tot
print("SIR total maks:", {m: max(v) for m, v in sir_tot.items()})
print("SIR saat ini maks:", {m: max(v) for m, v in sir_now.items()})
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for m in METHODS:
    axes[0].plot(sir_now[m], label=m, color=COLORS[m], linewidth=1.5)
    axes[1].plot(sir_tot[m], label=m, color=COLORS[m], linewidth=1.5)
axes[0].set_title("Node terinfeksi saat ini (SIR)")
axes[1].set_title("Total node pernah terinfeksi (SIR)")
for a in axes:
    a.set_xlabel("Iterasi"); a.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(IMG, "03_sir_curves.png")); plt.close(fig)

# 4. SIS: node 0 terinfeksi berapa kali
iters = run_sis()
node0 = sum(1 for it in iters if it["status"].get("0", 0) == 1)
print("node 0 terinfeksi pada", node0, "iterasi")
sis_curve = [it["node_count"][1] for it in iters]
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(sis_curve, color="#7C3AED", linewidth=2)
ax.set_xlabel("Iterasi"); ax.set_ylabel("Node terinfeksi")
ax.set_title("Model SIS: infeksi naik turun karena node bisa terinfeksi ulang")
fig.tight_layout(); fig.savefig(os.path.join(IMG, "04_sis_curve.png")); plt.close(fig)

import json
summary = {
    "nodes": G.number_of_nodes(),
    "edges": G.number_of_edges(),
    "threshold_final_infected": final,
    "threshold_least_method": min(final, key=final.get),
    "threshold_vs_ic": {"threshold": n_thr, "independent_cascade": n_ic,
                        "difference": n_thr - n_ic},
    "sir_total_max": {m: max(v) for m, v in sir_tot.items()},
    "sir_atpoint_max": {m: max(v) for m, v in sir_now.items()},
    "sir_most_total": max(sir_tot, key=lambda m: max(sir_tot[m])),
    "sir_most_atpoint": max(sir_now, key=lambda m: max(sir_now[m])),
    "sis_node0_infection_iterations": node0,
}
with open(os.path.join(BASE, "findings.json"), "w") as f:
    json.dump(summary, f, indent=2)
print("findings.json tersimpan")
print("SELESAI")
