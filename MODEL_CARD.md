# Model card

## Usage prevu

Tester l'existence d'un signal predictif reproductible dans les tirages FDJ et refuser une
prediction lorsque les preuves hors echantillon sont insuffisantes.

## Modeles

- Posterior bayesien beta-binomial fortement ramene vers `5/49`.
- Regression logistique L2 sur variables temporelles et effets par numero.
- Histogram Gradient Boosting avec profondeur et regularisation limitees.

Les probabilites marginales sont projetees sur la contrainte `somme = 5`. Le numero Chance reste
uniforme : son test global ne montre pas d'anomalie et les donnees ne justifient pas un modele a
dix classes supplementaire.

## Evaluation

- Decoupage chronologique strict et groupement des tirages d'une meme date.
- Selection des hyperparametres dans le passe de chaque fold externe.
- Score de Brier, log-loss, calibration et nombre de hits Top-5.
- Intervalle bootstrap, permutation appariee et correction de Holm.
- Qualification uniquement si l'intervalle du delta Brier est entierement meilleur que zero.

## Limites

- Seulement 2 859 tirages compatibles avec le format actuel.
- Aucun identifiant public de machine, jeu de boules, maintenance ou operateur dans les archives.
- Absence des combinaisons effectivement choisies par tous les joueurs.
- Changements possibles de materiel et de procedure non observables.
- Un avantage de score statistique ne garantirait pas un rendement superieur au prix des grilles.

## Resultat actuel

Aucun des trois modeles ne se qualifie. La sortie de production est `abstention`. Le mode force
sert uniquement au diagnostic et ne doit pas etre presente comme une prediction gagnante.
