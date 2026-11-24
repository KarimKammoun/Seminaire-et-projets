# TP : reconnaissance des émotions

**Nom : Karim Kammoun**  
**Date :** à compléter

## Objectif du TP

L'objectif est de reconnaître l'émotion associée à une séquence de visages en comparant ses points caractéristiques à ceux de séquences dont le label est connu. On compare une méthode de base à deux variantes : l'alignement spatial avec Procrustes et l'alignement temporel avec DTW.

## Données utilisées

Les données fournies contiennent 327 séquences, sept valeurs de labels et 118 personnes. Chaque trame est composée de 49 points à deux coordonnées. Le nombre de trames varie selon les séquences. Les fichiers utilisés ne documentent pas la correspondance entre les chiffres des labels et les noms des émotions ; on conserve donc les labels numériques.

## Question 1 : préparation des trames

La fonction `preforme` centre les points de chaque trame en soustrayant leur moyenne. Elle divise ensuite les coordonnées par la norme de Frobenius de la trame centrée. Cela réduit l'effet de la position et de la taille du visage. Chaque séquence devient ainsi une suite de formes normalisées.

## Question 2 : distance entre deux trames

Pour comparer deux trames normalisées X et Y, on calcule la distance angulaire d(X,Y) = arccos(<X,Y>). Le produit scalaire de Frobenius <X,Y> est la somme des produits des coordonnées correspondantes. La fonction `distance_trames` limite le résultat du produit scalaire à l'intervalle [-1, 1] avant d'appliquer arccos, afin d'éviter les erreurs d'arrondi. Une petite distance indique des formes proches.

## Question 3 : distance entre séquences et matrice

La fonction `distance_base` compare les trames de même indice, puis calcule la moyenne de leurs distances. Si les séquences ont des longueurs différentes, la comparaison s'arrête à la fin de la plus courte. La fonction `matrices_distances` construit une matrice pour chaque méthode : la case i,j contient la distance entre les séquences i et j. Une seule moitié est calculée car les distances utilisées sont symétriques.

## Question 4 : KNN et taux de reconnaissance

La fonction `evaluer` utilise KNN avec les distances préalablement calculées. Pour classer une séquence de test, elle cherche ses k voisins les plus proches parmi les séquences d'apprentissage et prédit leur label majoritaire. Le taux de reconnaissance est le nombre de prédictions correctes divisé par le nombre de séquences de test, multiplié par 100.

Pour cette expérience, k = 3. Les personnes sont séparées avec GroupShuffleSplit : environ 20 % des sujets pour le test, avec la graine 42. On obtient 259 séquences d'apprentissage et 68 de test. Une personne de test n'apparaît pas dans l'apprentissage. Les trois méthodes utilisent la même séparation.

## Question 5 : alignement spatial avec Procrustes

La fonction `distance_procrustes` utilise la fonction `procrustes` fournie dans le fichier du professeur, copiée dans `procrustes_prof.py` pour éviter d'exécuter le reste de son ancien script lors de l'import. Une comparaison de chaînes a été corrigée (`!=` au lieu de `is not`). Pour chaque paire de trames déjà normalisées, on aligne la seconde sur la première avec `scaling=False` et `reflection=False` : on autorise la rotation mais pas l'effet miroir ni une nouvelle mise à l'échelle. On calcule ensuite la distance angulaire moyenne des trames alignées.

## Question 6 : alignement temporel avec DTW

La fonction `distance_dtw` calcule les distances angulaires entre les trames, puis cherche le chemin de coût cumulé minimal qui aligne les deux séquences. Cela permet de comparer des expressions produites à des vitesses différentes. Ici DTW est évalué seul, sans Procrustes. Comme dans la fonction fournie, le coût est une somme non normalisée et peut donc dépendre de la longueur des séquences.

## Résultats et interprétation

| Méthode | Prédictions correctes | Taux | Écart avec la base |
|---|---:|---:|---:|
| Base | 38 / 68 | 55,88 % | — |
| Procrustes | 42 / 68 | 61,76 % | +5,88 points |
| DTW seul | 48 / 68 | 70,59 % | +14,71 points |

Pour cette séparation, les deux variantes améliorent le taux observé. Cela ne prouve pas qu'elles seront toujours meilleures : le résultat dépend notamment du choix des personnes, de k et du protocole. Une validation croisée par sujet et une étude de plusieurs valeurs de k aideraient à vérifier la robustesse des résultats. Les échelles numériques du coût DTW cumulé et des moyennes des autres méthodes sont différentes ; les taux KNN, eux, sont comparés avec le même protocole.

## Conclusion

Ce TP montre comment représenter des séquences de visages par des formes normalisées et les utiliser pour prédire des labels d'émotions. Dans l'expérience réalisée, les alignements spatial et temporel améliorent le taux par rapport à la méthode de base. Ces résultats restent à confirmer sur d'autres répartitions des personnes.
