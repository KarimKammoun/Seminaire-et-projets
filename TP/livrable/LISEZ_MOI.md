# Installation et utilisation sur PC

## Télécharger

Sur GitHub, ouvre la branche `arena/ad5285ca-seminaire-et-projets`, clique sur **Code**, puis **Download ZIP**. Décompresse le fichier. Conserve les dossiers tels quels : les trois fichiers `.mat` sont dans `TP`, et le programme est dans `TP/livrable`.

## Windows : installer Python et les dépendances

Installe Python 3 depuis https://www.python.org/downloads/ si nécessaire. Lors de l'installation, coche « Add Python to PATH ».

Ouvre le dossier `TP/livrable` dans l'explorateur Windows. Clique dans la barre d'adresse, écris `cmd` et appuie sur Entrée. Dans la fenêtre noire :

```text
py --version
py -m pip install -r requirements.txt
```

La première commande vérifie Python ; la seconde installe NumPy, SciPy et scikit-learn. Si `py` n'est pas reconnu, essaie `python` à sa place. Pour isoler les dépendances, on peut utiliser un environnement virtuel Python.

## Lancer le TP

Dans le même terminal :

```text
py tp_emotions.py
```

Le calcul crée ou remplace les résultats dans `TP/livrable/resultats/`. Les fichiers `.mat` doivent rester dans le dossier parent `TP`. Pour un essai rapide, tape `py tp_emotions.py --limite 30` ; cet essai est enregistré séparément et ne représente pas le résultat sur toutes les données. Pour changer le nombre de voisins : `py tp_emotions.py --k 5`.

Sur macOS ou Linux, remplace `py` par `python3` dans les commandes.

## Que lire ?

- `rapport.md` : méthode et résultats ; complète la date, relis le rapport et exporte-le en PDF si demandé.
- `resultats/resultats.txt` : taux et paramètres de l'expérience.
- `resultats/predictions_*.csv` : vrais labels et prédictions pour chaque séquence de test.
- `resultats/confusion_*.csv` : matrices de confusion ; lignes = vraies classes, colonnes = classes prédites dans l'ordre de l'en-tête.
- `resultats/matrices_distances.npz` : matrices de distances complètes lisibles avec NumPy.

Vérifie les consignes du professeur, comprends le code et indique l'aide utilisée si les règles de ton établissement le demandent.
