# TP — Reconnaissance d'émotions à partir de séquences de visages

**Nom :** Karim Kammoun  
**Date :** à compléter

## Objectif

Dans ce TP, on cherche à reconnaître une émotion à partir d'une séquence de visages. Pour cela, on compare les points du visage avec ceux de séquences dont le label est connu. On teste d'abord une méthode simple, puis on regarde si les alignements spatial et temporel permettent d'améliorer le résultat.

## Données

Le jeu fourni contient 327 séquences, sept valeurs de labels et 118 identifiants de personnes. Une trame est formée de 49 points à deux coordonnées. Les séquences ne possèdent pas toutes le même nombre de trames. Les fichiers ne documentent pas ici la correspondance entre les chiffres des labels et les noms d'émotions : on conserve donc les labels numériques.

## Prétraitement et distance entre trames

On centre chaque trame en soustrayant la moyenne de ses points, puis on la divise par sa norme de Frobenius. Pour deux trames normalisées X et Y, on utilise la distance angulaire arccos(<X,Y>), où le produit scalaire est la somme des produits des coordonnées correspondantes. On limite numériquement le produit scalaire à [-1, 1].

## Comparaison des séquences

- **Base :** moyenne des distances entre trames de même indice, jusqu'à la fin de la séquence la plus courte.
- **Procrustes :** pour chaque paire de trames correspondantes, on cherche la rotation 2D optimale sans réflexion. Les trames sont déjà centrées et normalisées. On calcule ensuite la moyenne des distances angulaires. La rotation optimale est calculée directement par une formule 2D, équivalente à un alignement orthogonal sans réflexion.
- **DTW :** on utilise les distances angulaires comme coûts locaux et on cherche le coût cumulé minimal d'un alignement temporel. Ici DTW est utilisé sans Procrustes. Le coût final est une somme non normalisée, comme dans la fonction DTW fournie ; il peut donc dépendre de la longueur des séquences. Il ne faut pas comparer directement son échelle numérique aux moyennes des autres méthodes.

Une matrice de distances complète est calculée pour chaque méthode. KNN utilise la matrice de distances pour faire voter les k séquences d'apprentissage les plus proches. Une séquence de test ne peut pas être son propre voisin.

## Protocole expérimental

Les personnes ont été séparées entre apprentissage et test avec `GroupShuffleSplit` : 20 % des sujets pour le test, graine 42. Cela donne 259 séquences d'apprentissage et 68 séquences de test ; une même personne ne figure pas dans les deux ensembles. KNN utilise k = 3. Les trois méthodes sont évaluées sur exactement les mêmes données. Le score est la proportion de prédictions correctes parmi les 68 séquences de test.

## Résultats obtenus sur les fichiers fournis

| Méthode | Prédictions correctes | Taux | Écart avec la base |
|---|---:|---:|---:|
| Base | 38 / 68 | 55,88 % | — |
| Procrustes | 42 / 68 | 61,76 % | +5,88 points |
| DTW seul | 48 / 68 | 70,59 % | +14,71 points |

Les écarts sont arrondis à deux décimales. Pour cette séparation des données, les deux variantes donnent de meilleurs résultats que la méthode de base. Cela ne veut pas dire qu'elles seront toujours meilleures : le résultat peut changer selon les personnes choisies, la valeur de k ou le protocole. Le coût DTW non normalisé peut favoriser certaines longueurs. Une validation croisée par sujet et une étude du choix de k permettraient une évaluation plus robuste.

## Conclusion

La géométrie du visage et son évolution temporelle peuvent être utilisées pour comparer des expressions. Dans cette expérience, l'alignement spatial et surtout l'alignement temporel ont amélioré le taux observé. Il reste important de vérifier la reproductibilité et les limites du protocole avant de généraliser ces résultats.
