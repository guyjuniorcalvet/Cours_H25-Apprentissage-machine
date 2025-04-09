📚 Projet de session – 8IAR403 : Apprentissage automatique pour la science des données
🧾 Présentation
Ce dépôt regroupe les travaux pratiques réalisés dans le cadre du cours 8IAR403 – Apprentissage automatique pour la science des données offert à l’Université du Québec à Chicoutimi (session Hiver 2025). Le projet repose sur une étude de cas portant sur un site de vente en ligne, et suit l'ensemble du cycle de vie d’un projet en apprentissage automatique : de la préparation des données jusqu’à l’entraînement de modèles avancés comme les réseaux de neurones convolutifs.

🧱 Structure du dépôt
bash
Copier
Modifier
📁 devoir1/
    └─ devoir1_preparation_donnees.ipynb
📁 devoir2/
    └─ devoir2_modeles_classiques.ipynb
📁 devoir3/
    ├─ devoir3_mlp_vente_en_ligne.ipynb
    ├─ devoir3_cnn_fashion_mnist.ipynb
📁 data/
    ├─ Customer.csv
    ├─ CountryGDP.csv
    └─ CountryPopulation.csv
README.md
requirements.txt
🧠 Contenu des travaux
📌 Devoir 1 – Compréhension et préparation des données
Nettoyage des données : traitement des valeurs manquantes, aberrantes et types inappropriés.

Enrichissement du jeu de données client (Customer.csv) par jointure avec des indicateurs macroéconomiques (CountryGDP.csv, CountryPopulation.csv).

Conception de pipelines de transformation automatisés avec scikit-learn, incluant des FunctionTransformer et des traitements personnalisés.

Objectif : obtenir une Table de Base d’Apprentissage (TBA) prête à l’entraînement.

📌 Devoir 2 – Entraînement de modèles classiques (arbres de décision)
Transformation de la variable cible (revenu) en classes binaires et multi-classes.

Entraînement de plusieurs modèles DecisionTreeClassifier selon différents échantillons et configurations de données.

Validation croisée (k=3 et k=10), réglage des hyperparamètres (profondeur, taille des nœuds) avec GridSearchCV.

Évaluation des modèles avec les métriques : précision, rappel, F1-score.

Représentation graphique des performances selon :

Taille de l’échantillon

Présence ou absence des variables socioéconomiques (PIB, population)

Paramètres optimisés

Temps d'entraînement

📌 Devoir 3 – Réseaux de neurones (MLP et CNN)
🔹 Partie 1 – Perceptron Multi-Couches (MLP) sur les données clients
Utilisation de MLPClassifier pour entraîner un modèle binaire et un modèle multi-classes.

Optimisation par validation croisée et recherche aléatoire (RandomizedSearchCV) des hyperparamètres :

Nombre de couches cachées

Nombre de neurones

Fonction d’activation

Solveur

Taux d’apprentissage

Évaluation via courbes d’apprentissage et comparaison avec les modèles du Devoir 2.

🔹 Partie 2 – CNN pour la classification d’images (Fashion MNIST)
Entraînement d’un réseau CNN personnalisé sur le dataset fashion_mnist.

Amélioration des performances par :

Augmentation de données (ImageDataGenerator)

Apprentissage par transfert avec des modèles pré-entraînés comme VGG16

Étude comparative des modèles (PMC-Keras, CNN simple, CNN avec augmentation, CNN pré-entraîné) selon :

Exactitude (accuracy)

Temps d’entraînement

⚙️ Installation et exécution
1. Cloner le dépôt
bash
Copier
Modifier
git clone https://github.com/ton-utilisateur/8IAR403-projet-session.git
cd 8IAR403-projet-session
2. Créer un environnement virtuel (recommandé)
bash
Copier
Modifier
python -m venv env
source env/bin/activate     # Sur Linux/Mac
env\Scripts\activate.bat    # Sur Windows
3. Installer les dépendances
bash
Copier
Modifier
pip install -r requirements.txt
4. Lancer les notebooks
Tu peux ouvrir les fichiers .ipynb avec JupyterLab, Google Colab, ou VSCode avec l’extension Python.

📦 Technologies utilisées
Python (>=3.9)

Pandas, NumPy, Scikit-learn

Matplotlib, Seaborn

TensorFlow, Keras

Jupyter Notebook

👨‍🎓 Auteur
Nom : [Ton nom complet ici]
*Étudiant à l’UQAC – Baccalauréat en informatique, science des données et intelligence des affaires
Cours : 8IAR403 – Apprentissage automatique
Enseignant : Pr. Abdenour Bouzouane
Session : Hiver 2025
