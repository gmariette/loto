# Model card

## Usage prevu

Tester l'existence d'un signal predictif reproductible dans les tirages FDJ et refuser une
prediction lorsque les preuves hors echantillon sont insuffisantes.

## Modeles

- Posterior bayesien beta-binomial fortement ramene vers `5/49`.
- Regression logistique L2 sur variables temporelles et effets par numero.
- Histogram Gradient Boosting avec profondeur et regularisation limitees.
- Regression Ridge et gradient boosting pour le volume de grilles.

Les probabilites marginales sont projetees sur la contrainte `somme = 5`, puis retractees vers
l'uniforme avec un poids choisi dans le passe. Le numero Chance reste
uniforme : son test global ne montre pas d'anomalie et les donnees ne justifient pas un modele a
dix classes supplementaire.

## Evaluation

- Decoupage chronologique strict et groupement des tirages d'une meme date.
- Selection des hyperparametres dans le passe de chaque fold externe.
- Score de Brier, log-loss, calibration et nombre de hits Top-5.
- Intervalle bootstrap, permutation appariee et correction de Holm.
- Qualification uniquement si l'intervalle du delta Brier est entierement meilleur que zero.
- Participation inferee par le rang 9 et comparee a une mediane segmentee sur des folds futurs.

## Limites

- Seulement 2 859 tirages compatibles avec le format actuel.
- Aucun identifiant public de machine, jeu de boules, maintenance ou operateur dans les archives.
- Absence des combinaisons effectivement choisies par tous les joueurs.
- Le volume de grilles est un proxy statistique, pas une mesure FDJ certifiee.
- Changements possibles de materiel et de procedure non observables.
- Un avantage de score statistique ne garantirait pas un rendement superieur au prix des grilles.

## Resultat actuel

Aucun des trois modeles de numeros ne se qualifie. La sortie de production est `abstention`. Le
gradient boosting de participation se qualifie contre sa reference et sert uniquement au calcul
de partage et de valeur. Le mode force ne doit pas etre presente comme une prediction gagnante.
