#!/usr/bin/env python3
"""TP : reconnaissance des émotions à partir de séquences de visages."""
from pathlib import Path
import argparse
import csv
import time

import numpy as np
from scipy.io import loadmat
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from procrustes_prof import procrustes

DOSSIER = Path(__file__).resolve().parent
DONNEES = DOSSIER.parent


def charger_donnees():
    """Les fichiers MATLAB sont dans TP, le dossier parent de livrable."""
    sequences = loadmat(DONNEES / 'sequences_ckp.mat')['sequences'].ravel()
    labels = loadmat(DONNEES / 'labels_ckp.mat')['labels'].ravel()
    sujets = loadmat(DONNEES / 'subject_labels_ckp.mat')['subject_labels'].ravel()
    if not (len(sequences) == len(labels) == len(sujets)):
        raise ValueError('Le nombre de séquences, de labels et de sujets doit être identique.')
    return sequences, labels, sujets


# QUESTION 1 : centrer et normaliser chaque trame avec la norme de Frobenius.
def preforme(sequence):
    # Passer de (points, temps, coordonnées) à (temps, points, coordonnées).
    trames = np.asarray(sequence, dtype=float).transpose(1, 0, 2)
    trames_centrees = trames - trames.mean(axis=1, keepdims=True)
    normes = np.linalg.norm(trames_centrees, axis=(1, 2), keepdims=True)
    if np.any(normes < 1e-12):
        raise ValueError('Impossible de normaliser une trame de norme nulle.')
    return trames_centrees / normes


# QUESTION 2 : distance angulaire entre deux trames normalisées.
def distance_trames(x, y):
    produit_scalaire = np.sum(x * y)
    return float(np.arccos(np.clip(produit_scalaire, -1.0, 1.0)))


def couts_trames(sequence_a, sequence_b):
    """Distances entre chaque paire de trames, utilisées par DTW."""
    a = sequence_a.reshape(len(sequence_a), -1)
    b = sequence_b.reshape(len(sequence_b), -1)
    produits = a @ b.T
    return np.arccos(np.clip(produits, -1.0, 1.0))


# QUESTION 3 : distance moyenne entre trames de même indice et matrice complète.
def distance_base(sequence_a, sequence_b):
    nombre_trames = min(len(sequence_a), len(sequence_b))
    distances = [distance_trames(sequence_a[i], sequence_b[i])
                 for i in range(nombre_trames)]
    return float(np.mean(distances))


def matrices_distances(sequences):
    nombre = len(sequences)
    matrices = {nom: np.zeros((nombre, nombre)) for nom in ('base', 'procrustes', 'dtw')}
    fonctions = {'base': distance_base, 'procrustes': distance_procrustes, 'dtw': distance_dtw}
    debut = time.monotonic()
    for i in range(nombre):
        if i % 20 == 0:
            print(f'Distances : {i}/{nombre} lignes ({time.monotonic() - debut:.0f} s)', flush=True)
        # Les distances sont symétriques : calculer une moitié suffit.
        for j in range(i + 1, nombre):
            for nom, fonction in fonctions.items():
                distance = fonction(sequences[i], sequences[j])
                matrices[nom][i, j] = distance
                matrices[nom][j, i] = distance
    return matrices


# QUESTION 5 : aligner les trames avec la fonction Procrustes du professeur.
def distance_procrustes(sequence_a, sequence_b):
    nombre_trames = min(len(sequence_a), len(sequence_b))
    distances = []
    for i in range(nombre_trames):
        x, y = sequence_a[i], sequence_b[i]
        # Les trames sont déjà centrées et normalisées. On autorise la rotation,
        # mais pas le changement d'échelle ni la réflexion (effet miroir).
        _, y_aligne, _ = procrustes(x, y, scaling=False, reflection=False)
        distances.append(distance_trames(x, y_aligne))
    return float(np.mean(distances))


# QUESTION 6 : aligner les séquences dans le temps avec DTW.
def distance_dtw(sequence_a, sequence_b):
    couts = couts_trames(sequence_a, sequence_b)
    # Chaque case contient le meilleur coût cumulé pour atteindre cette paire.
    precedent = np.full(len(sequence_b) + 1, np.inf)
    precedent[0] = 0.0
    for ligne in couts:
        actuel = np.full(len(sequence_b) + 1, np.inf)
        for j, cout in enumerate(ligne, start=1):
            meilleur_precedent = min(precedent[j], actuel[j - 1], precedent[j - 1])
            actuel[j] = cout + meilleur_precedent
        precedent = actuel
    # Comme dans le code fourni, ce coût est une somme, non une moyenne.
    return float(precedent[-1])


# QUESTION 4 : classification KNN et calcul du taux de reconnaissance.
def evaluer(matrices, labels, sujets, train, test, k, dossier):
    dossier.mkdir(exist_ok=True)
    np.savez_compressed(dossier / 'matrices_distances.npz', **matrices)
    resultats = []
    classes = np.unique(labels)

    for nom, matrice in matrices.items():
        modele = KNeighborsClassifier(n_neighbors=k, metric='precomputed')
        modele.fit(matrice[np.ix_(train, train)], labels[train])
        predictions = modele.predict(matrice[np.ix_(test, train)])
        taux = 100 * accuracy_score(labels[test], predictions)
        resultats.append((nom, taux))
        print(f'{nom:12s} : {taux:.2f} %')

        confusion = confusion_matrix(labels[test], predictions, labels=classes)
        np.savetxt(dossier / f'confusion_{nom}.csv', confusion, delimiter=',', fmt='%d',
                   header=','.join(map(str, classes)), comments='')
        with (dossier / f'predictions_{nom}.csv').open('w', newline='', encoding='utf-8') as fichier:
            ecrivain = csv.writer(fichier)
            ecrivain.writerow(['indice_sequence', 'sujet', 'label_reel', 'label_predit'])
            ecrivain.writerows(zip(test, sujets[test], labels[test], predictions))

    with (dossier / 'resultats.txt').open('w', encoding='utf-8') as fichier:
        fichier.write(f'Séquences : {len(labels)} ; apprentissage : {len(train)} ; test : {len(test)}\n')
        fichier.write(f'Séparation par sujets : GroupShuffleSplit, test_size=0.2, random_state=42 ; k={k}\n')
        fichier.write('DTW : coût cumulé non normalisé ; Procrustes : rotation sans réflexion.\n')
        for nom, taux in resultats:
            ecart = taux - resultats[0][1]
            fichier.write(f'{nom} : {taux:.2f} % (écart avec base : {ecart:+.2f} points)\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--k', type=int, default=3, help='Nombre de voisins (défaut : 3)')
    parser.add_argument('--limite', type=int, help='Nombre de séquences pour un essai rapide')
    args = parser.parse_args()
    sequences, labels, sujets = charger_donnees()
    if args.limite is not None:
        if args.limite < 10 or args.limite > len(sequences):
            parser.error('--limite doit être comprise entre 10 et le nombre total de séquences')
        sequences = sequences[:args.limite]
        labels = labels[:args.limite]
        sujets = sujets[:args.limite]

    sequences = [preforme(sequence) for sequence in sequences]
    separation = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train, test = next(separation.split(sequences, labels, groups=sujets))
    if args.k < 1 or args.k > len(train):
        parser.error('--k doit être compris entre 1 et le nombre de séquences d’apprentissage')
    print(f'{len(sequences)} séquences ; apprentissage : {len(train)}, test : {len(test)} ; k={args.k}')
    print('Séparation par personne, graine 42. Calcul des matrices complètes...')

    matrices = matrices_distances(sequences)
    dossier = DOSSIER / ('resultats_essai' if args.limite else 'resultats')
    evaluer(matrices, labels, sujets, train, test, args.k, dossier)
    print(f'Fichiers enregistrés dans {dossier}')


if __name__ == '__main__':
    main()
