# flask-ml-diabetes
Api permettant de définir si un individu est diabétique en fonction de données sur son sang.

# Utilisation
`pip -r requirements.txt` pour télécharger les dépendances.

Puis `python api.py` pour lancer l'application web.

# Modèle utilisé
Le modèle le plus convaincant a été une régression logistique.

# Paramètres de l'API
- num_preg
- glucose_conc
- diastolic_bp
- thicness
- insulin
- bmi
- diab_pred
- age

# Réponse de l'API
- resultat: Pourcentage d'avoir du diabète
- accuracy: accuracy score du modèle utilisé