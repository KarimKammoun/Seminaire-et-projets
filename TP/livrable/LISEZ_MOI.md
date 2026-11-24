# TP reconnaissance d'émotions — installation sur ton PC

Ce dossier est autonome : il contient le programme, les trois fichiers de données et les résultats obtenus lors d'une exécution. Le programme fonctionne avec Python 3 et les bibliothèques de `requirements.txt`.

## 1. Décompresser et installer Python

Décompresse l'archive dans un dossier, par exemple `TP_Emotions`. Installe Python 3 depuis https://www.python.org/downloads/ si nécessaire. Sous Windows, coche « Add Python to PATH » lors de l'installation. Avec Anaconda, tu peux également utiliser l'Anaconda Prompt.

## 2. Installer les bibliothèques

Ouvre un terminal dans le dossier décompressé. Sur Windows, tu peux ouvrir le dossier dans l'Explorateur, écrire `cmd` dans la barre d'adresse et appuyer sur Entrée. Exécute :

```text
py -m pip install -r requirements.txt
```

Sur macOS ou Linux, utilise plutôt :

```text
python3 -m pip install -r requirements.txt
```

Si `py` ne marche pas sous Windows, essaie `python` à sa place. Si nécessaire, installe les bibliothèques dans un environnement virtuel.

## 3. Lancer le TP

Dans le même terminal :

```text
py tp_emotions.py
```

Sur macOS ou Linux :

```text
python3 tp_emotions.py
```

Les fichiers `.mat` doivent rester à côté de `tp_emotions.py`. L'exécution calcule les trois matrices complètes, puis crée ou remplace les fichiers du dossier `resultats`. Cela peut prendre plus ou moins de temps selon le PC. Pour un essai rapide seulement :

```text
py tp_emotions.py --limite 30
```

Les résultats de cet essai sont enregistrés séparément dans `resultats_essai` et ne doivent pas être présentés comme les résultats sur toutes les données. Pour changer k : `py tp_emotions.py --k 5`.

## 4. Fichiers et résultats

- `tp_emotions.py` : programme à lire, comprendre et exécuter.
- `rapport.md` : rapport explicatif modifiable. Il peut être ouvert dans un éditeur de texte et exporté en PDF, par exemple avec Word ou LibreOffice après copie du contenu.
- `resultats/resultats.txt` : taux et paramètres de l'exécution.
- `resultats/predictions_*.csv` : étiquettes réelles et prédites des séquences de test ; indices Python à partir de 0.
- `resultats/confusion_*.csv` : matrices de confusion ; lignes = vraies classes et colonnes = classes prédites, dans l'ordre de l'en-tête.
- `resultats/matrices_distances.npz` : les trois matrices complètes, lisibles avec NumPy (`np.load`).

Lis et adapte le rapport, comprends le code et vérifie les consignes du professeur avant de remettre ton travail. Ne présente pas comme tien un travail que tu ne comprends pas ; signale l'aide utilisée si les règles de ton établissement le demandent.
