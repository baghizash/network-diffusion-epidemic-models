# Model Difusi Epidemi pada Jaringan

Simulasi model difusi memakai ndlib: Threshold, Independent Cascade, SIR, dan
SIS pada jaringan nyata, plus analisis pengaruh pemilihan seed node terhadap
luas penyebaran.

Proyek ini adalah Module 4 dari kursus Network Modeling and Analysis in Python
(University of Michigan).

## Data

Folder `data` berisi satu jaringan nyata:

* `german.txt`: edgelist jaringan dengan 1168 node dan 1243 edge.

## Yang dikerjakan

1.  Model Threshold dengan 5 metode pemilihan seed (50 iterasi, 5 seed,
    threshold 0.3). Seed acak menghasilkan infeksi paling sedikit (36 node),
    jauh di bawah degree (298), pagerank (298), betweenness (170), dan
    closeness (135). Lihat `charts/01_threshold_seeds.png`.
2.  Perbandingan Threshold vs Independent Cascade (seed degree): 1038 lawan 97
    node terinfeksi, selisih 941. Model Threshold menyebar jauh lebih luas.
    Lihat `charts/02_threshold_vs_ic.png`.
3.  Model SIR (beta 0.1, gamma 0.05): total node pernah terinfeksi tertinggi
    dengan seed degree (412), sedangkan puncak infeksi pada satu waktu
    tertinggi dengan seed pagerank (149). Lihat `charts/03_sir_curves.png`.
4.  Model SIS: node bisa terinfeksi berulang. Node 0 tercatat terinfeksi pada
    8 iterasi. Lihat `charts/04_sis_curve.png`.

## Cara menjalankan

```bash
python analyze_diffusion.py
```

Script ini menjalankan semua simulasi dengan seed 42 agar hasilnya
reproducible, menyimpan chart ke folder `charts`, dan menulis ringkasan angka
ke `findings.json`.

## Tools

Python, NetworkX, ndlib, pandas, NumPy, matplotlib.
