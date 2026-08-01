# Model card

## Usage prevu

Tester l'existence d'un signal predictif reproductible dans les tirages FDJ et refuser une
prediction lorsque les preuves hors echantillon sont insuffisantes.

## Modeles

- Posterior bayesien beta-binomial fortement ramene vers `5/49`.
- Posterior bayesien temporel sur fenetre 10/50/200 avec lissage selectionne dans le passe.
- Regression logistique L2 sur variables temporelles et effets par numero.
- Regression logistique de classement dont la regularisation maximise les hits internes.
- Histogram Gradient Boosting avec profondeur et regularisation limitees.
- Regression Ridge et gradient boosting pour le volume de grilles.

Les probabilites marginales sont projetees sur la contrainte `somme = 5`, puis retractees vers
l'uniforme avec un poids choisi dans le passe. Le numero Chance reste
uniforme : son test global ne montre pas d'anomalie et les donnees ne justifient pas un modele a
dix classes supplementaire.

## Evaluation

- Decoupage chronologique strict et groupement des tirages d'une meme date.
- Entrainement et evaluation isoles par jeu cible.
- Selection des hyperparametres dans le passe de chaque fold externe.
- Graines stables par identite de modele, independantes de l'ordre des challengers.
- Score de Brier, log-loss, calibration et nombre de hits Top-5.
- Intervalle bootstrap, permutation appariee et correction de Holm.
- Qualification probabiliste par delta Brier ou qualification de classement par gain Top-5.
- Null Top-5 hypergeometrique exact et Holm conjoint sur les modeles et les deux metriques.
- Participation inferee par le rang 9 et comparee a une mediane segmentee sur des folds futurs.
- Biais de retransformation mesure en niveau et facteur de smearing estime sans donnees futures.
- Backtest de valeur walk-forward comparant MAE/RMSE, couverture et decisions a une reference naive.
- Inference du moteur de valeur par blocs contigus de 12 tirages pour conserver la dependance locale.
- Reentrainement toutes les 52 dates et calibration interne de l'horizon et des queues predictives.
- Registre prospectif append-only, chaine SHA-256 et scoring seulement apres publication du bareme.
- Bundle JSON autonome: verification des chaines et recalcul des metriques depuis chaque bareme.
- Empreintes SHA-256 des fichiers et du snapshot logique; source FDJ incluse dans chaque score v2.
- Benchmark naif fige avant tirage et qualification par empreinte sur une cohorte fixe de 100 scores.
- Identite scientifique SHA-256 du code, des parametres et des dependances du moteur de valeur.
- Cycle idempotent pilote par manifeste, export atomique et journal d'execution chaine.

## Limites

- Seulement 2 859 tirages compatibles avec le format actuel.
- Aucun identifiant public de machine, jeu de boules, maintenance ou operateur dans les archives.
- Absence des combinaisons effectivement choisies par tous les joueurs.
- Le volume de grilles est un proxy statistique, pas une mesure FDJ certifiee.
- Les residus empiriques de participation restent globaux et ne capturent pas toute
  heteroscedasticite conditionnelle.
- Le gradient boosting n'extrapole pas naturellement au-dela des jackpots d'apprentissage.
- La couverture observee de 93,90 % reste legerement sous la cible nominale de 95 %.
- Changements possibles de materiel et de procedure non observables.
- Un avantage de score statistique ne garantirait pas un rendement superieur au prix des grilles.
- Une chaine locale ne prouve sa chronologie que si son hash est publie avant le tirage.
- Le bundle prouve la coherence interne, pas l'authenticite du bareme; sa source officielle reste
  indispensable.
- Un hash de fichier prouve l'identite d'une entree, pas sa qualite ni son exhaustivite.
- Cent scores correspondent a un protocole pre-enregistre, pas a une garantie de stabilite future;
  les observations ulterieures sont publiees separement en surveillance.
- L'empreinte separe les changements declares; elle ne prouve pas a elle seule que l'environnement
  d'execution n'a pas ete falsifie.
- L'automatisation evite les doublons mais ne peut pas inventer un jackpot ou une source officielle;
  un manifeste incorrect reste refuse ou produit une entree incorrecte mais tracable.
- Le journal d'execution reste une preuve locale tant que sa tete n'est pas publiee exterieurement.
- Une seule prevision prospective est actuellement en attente; aucune performance future ne peut
  encore etre estimee.

## Resultat actuel

Aucun des cinq modeles de numeros ne se qualifie. La sortie de production est `abstention`. Le
gradient boosting de participation se qualifie contre sa reference et sert uniquement au calcul
de partage et de valeur. Le mode force ne doit pas etre presente comme une prediction gagnante.
