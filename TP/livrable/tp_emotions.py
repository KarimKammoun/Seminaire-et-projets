#!/usr/bin/env python3
"""TP de reconnaissance d'émotions : formes, Procrustes, DTW et KNN."""
from pathlib import Path
import argparse
import csv
import time
import numpy as np
from scipy.io import loadmat
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import KNeighborsClassifier

DOSSIER = Path(__file__).resolve().parent

def charger_donnees():
    sequences = loadmat(DOSSIER / 'sequences_ckp.mat')['sequences'].ravel()
    labels = loadmat(DOSSIER / 'labels_ckp.mat')['labels'].ravel()
    sujets = loadmat(DOSSIER / 'subject_labels_ckp.mat')['subject_labels'].ravel()
    if not (len(sequences) == len(labels) == len(sujets)):
        raise ValueError('Les nombres de séquences, labels et sujets diffèrent.')
    return sequences, labels, sujets

def preforme(sequence):
    # Entrée : points × temps × coordonnées. Sortie : temps × points × coordonnées.
    frames = np.asarray(sequence, dtype=float).transpose(1, 0, 2)
    frames = frames - frames.mean(axis=1, keepdims=True)
    normes = np.linalg.norm(frames, axis=(1, 2), keepdims=True)
    if np.any(normes < 1e-12):
        raise ValueError('Une trame a une norme nulle après centrage.')
    return frames / normes

def couts_frames(a, b):
    produits = a.reshape(len(a), -1) @ b.reshape(len(b), -1).T
    return np.arccos(np.clip(produits, -1, 1))

def distance_base(a, b):
    n = min(len(a), len(b))
    produits = np.einsum('tij,tij->t', a[:n], b[:n])
    return float(np.mean(np.arccos(np.clip(produits, -1, 1))))

def distance_procrustes(a, b):
    # Rotation optimale SO(2), sans réflexion ni redimensionnement supplémentaire.
    # Pour deux préformes, maximiser le produit scalaire revient à minimiser
    # la distance Procrustes. Équivalent à un alignement orthogonal par SVD.
    n = min(len(a), len(b))
    x, y = a[:n], b[:n]
    c = np.sum(x[:, :, 0]*y[:, :, 0] + x[:, :, 1]*y[:, :, 1], axis=1)
    s = np.sum(x[:, :, 0]*y[:, :, 1] - x[:, :, 1]*y[:, :, 0], axis=1)
    produits_optimaux = np.sqrt(c*c + s*s)
    return float(np.mean(np.arccos(np.clip(produits_optimaux, -1, 1))))

def distance_dtw(a, b):
    # Coût cumulé minimal, tel que la fonction dtw fournie dans le TP.
    couts = couts_frames(a, b)
    previous = np.full(len(b) + 1, np.inf)
    previous[0] = 0.0
    for ligne in couts:
        current = np.full(len(b) + 1, np.inf)
        for j, cost in enumerate(ligne, 1):
            current[j] = cost + min(previous[j], current[j-1], previous[j-1])
        previous = current
    return float(previous[-1])

def matrices_distances(sequences):
    n = len(sequences)
    matrices = {nom: np.zeros((n, n)) for nom in ('base', 'procrustes', 'dtw')}
    debut = time.monotonic()
    for i in range(n):
        if i % 20 == 0:
            print(f'Distances : {i}/{n} lignes ({time.monotonic()-debut:.0f} s)', flush=True)
        for j in range(i + 1, n):
            for nom, fonction in [('base', distance_base), ('procrustes', distance_procrustes), ('dtw', distance_dtw)]:
                d = fonction(sequences[i], sequences[j])
                matrices[nom][i, j] = matrices[nom][j, i] = d
    return matrices

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--k', type=int, default=3, help='Nombre de voisins (défaut : 3)')
    parser.add_argument('--limite', type=int, help='Nombre de séquences pour un essai rapide, pas pour le résultat final')
    args = parser.parse_args()
    sequences, labels, sujets = charger_donnees()
    if args.limite is not None:
        if args.limite < 10 or args.limite > len(sequences):
            parser.error('--limite doit être comprise entre 10 et le nombre de séquences')
        sequences, labels, sujets = sequences[:args.limite], labels[:args.limite], sujets[:args.limite]
    sequences = [preforme(s) for s in sequences]
    train, test = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(sequences, labels, groups=sujets))
    if args.k < 1 or args.k > len(train):
        parser.error('--k doit être entre 1 et la taille de l’ensemble d’apprentissage')
    print(f'{len(sequences)} séquences ; apprentissage : {len(train)}, test : {len(test)} ; k={args.k}')
    print('Séparation par personne, graine 42. Calcul des matrices complètes...')
    matrices = matrices_distances(sequences)
    dossier = DOSSIER / ('resultats_essai' if args.limite else 'resultats')
    dossier.mkdir(exist_ok=True)
    np.savez_compressed(dossier / 'matrices_distances.npz', **matrices)
    lignes = []
    for nom, matrice in matrices.items():
        knn = KNeighborsClassifier(n_neighbors=args.k, metric='precomputed')
        knn.fit(matrice[np.ix_(train, train)], labels[train])
        prediction = knn.predict(matrice[np.ix_(test, train)])
        taux = 100 * accuracy_score(labels[test], prediction)
        lignes.append((nom, taux))
        classes = np.unique(labels)
        confusion = confusion_matrix(labels[test], prediction, labels=classes)
        np.savetxt(dossier / f'confusion_{nom}.csv', confusion, delimiter=',', fmt='%d',
                   header=','.join(map(str, classes)), comments='')
        with (dossier / f'predictions_{nom}.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f); writer.writerow(['indice_sequence', 'sujet', 'label_reel', 'label_predit'])
            writer.writerows(zip(test, sujets[test], labels[test], prediction))
        print(f'{nom:12s} : {taux:.2f} %')
    with (dossier / 'resultats.txt').open('w', encoding='utf-8') as f:
        f.write(f'Séquences : {len(sequences)} ; apprentissage : {len(train)} ; test : {len(test)}\n')
        f.write(f'Séparation par sujets : GroupShuffleSplit, test_size=0.2, random_state=42 ; k={args.k}\n')
        f.write('DTW : coût cumulé non normalisé ; Procrustes : rotation sans réflexion.\n')
        for nom, taux in lignes:
            f.write(f'{nom} : {taux:.2f} % (écart avec base : {taux-lignes[0][1]:+.2f} points)\n')
    print(f'Fichiers enregistrés dans {dossier}')

if __name__ == '__main__':
    main()
