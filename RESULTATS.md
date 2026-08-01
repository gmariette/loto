# Resultats sur toutes les archives

Calculs effectues le 1er aout 2026 avec les 11 archives disponibles sur la page historique FDJ.
La base locale contient 5 566 tirages et 40 660 lignes de gains par rang.

## Regime actuel : 5/49 + Chance

Echantillon : 2 859 tirages du 6 octobre 2008 au 29 juillet 2026, dont 2 788 Loto,
60 Super Loto et 11 Grand Loto.

| Test | Observation | Reference hasard | Resultat |
|---|---:|---:|---:|
| Uniformite des 49 numeros | chi2 = 32,47 | Monte-Carlo, 5 000 repetitions | p = 0,910 |
| Uniformite de Chance | chi2 = 8,06 | Monte-Carlo, 5 000 repetitions | p = 0,525 |
| Numeros communs entre deux tirages | 0,5098 | 25/49 = 0,5102 | ecart = -0,0004 |

La paire la plus atypique est `7-11`, observee 46 fois contre 24,31 attendues. Elle a ete choisie
apres examen de 1 176 paires. Son test binomial bilateral corrige par Bonferroni donne environ
`p = 0,123` : l'ecart ne franchit pas le seuil global de 5 % et ne constitue pas une prediction.

## Backtest chronologique

Chaque prediction n'utilise que les tirages de dates strictement anterieures. Les 500 premiers
tirages servent de minimum d'apprentissage, puis 2 359 tirages sont evalues.

| Modele | Delta Brier contre uniforme | p bilateral | Top-5 moyen |
|---|---:|---:|---:|
| Frequence lissee, alpha 20 | +0,0000645 | 1,03e-7 | 0,5214 |
| Frequence decroissante, demi-vie 50 | +0,0002650 | 1,35e-24 | 0,5100 |
| Frequence decroissante, demi-vie 200 | +0,0001452 | 8,39e-16 | 0,5100 |

Un delta positif est moins bon. Les trois modeles historiques sont donc significativement moins
bien calibres que la probabilite uniforme `5/49`. Le leger surplus de Top-5 du premier modele ne
compense pas ses probabilites moins bien calibrees et n'implique aucun avantage monetaire.

## Validation ML imbriquee

La version 0.2 ajoute 63 variables, une selection interne d'hyperparametres, trois folds externes,
2 000 bootstraps et permutations, puis une correction de Holm entre modeles. L'evaluation porte
sur 2 309 tirages jamais utilises pour ajuster le fold correspondant.

| Modele | Delta Brier | IC 95 % du delta | Log-loss | Hits Top-5 | Qualifie |
|---|---:|---:|---:|---:|---:|
| Bayesien, prior 1000 | +0,0000326 | [+0,0000175 ; +0,0000467] | 0,32972 | 0,5193 | non |
| Logistique, C selectionne | +0,0001913 | [+0,0001403 ; +0,0002410] | 0,33062 | 0,5106 | non |
| Gradient boosting | +0,0000240 | [+0,0000091 ; +0,0000389] | 0,32968 | 0,5110 | non |

La log-loss uniforme vaut `0,32954`. Tous les intervalles sont strictement positifs : les modeles
ML sont moins bons que l'uniforme, pas seulement non concluants. La commande `ml-predict` renvoie
donc `abstention`. En mode force, la meilleure variante n'ecarte les probabilites marginales que
d'environ 10,2 % a 10,8 %, ce qui illustre l'absence de signal fort.

## Esperance monetaire

Sur 1 471 tirages Loto possedant les neuf rangs modernes, un jackpot annonce de 10 M EUR et zero
co-gagnant suppose donne une esperance estimee de `1,32 EUR` pour `2,20 EUR` mises, soit un ROI de
`59,8 %`. Le seuil ponctuel approche `26,86 M EUR`, avant modelisation des codes participants,
du partage reel du jackpot et des variations futures de rapports. La decision reste `no_bet`.

## Regime historique : 6/49 + complementaire

Echantillon separe : 2 707 tirages du 3 janvier 1996 au 4 octobre 2008, dont 2 664 Loto et
43 Super Loto.

| Test | Observation | Reference hasard | Resultat |
|---|---:|---:|---:|
| Uniformite des 49 numeros | chi2 = 33,84 | Monte-Carlo, 5 000 repetitions | p = 0,854 |
| Numeros communs entre deux tirages | 0,7180 | 36/49 = 0,7347 | ecart = -0,0167 |

Ce regime confirme l'absence d'anomalie globale, mais ne peut pas entrainer un predicteur du jeu
actuel car le nombre de boules et le numero special sont differents.

## Verdict

Les archives completes ne fournissent aucun signal global reproductible permettant de battre le
modele uniforme. Augmenter le volume historique renforce ce verdict : les heuristiques de
frequence se degradent hors echantillon. Le seul usage defensable de l'algorithme est donc l'audit
continu, la diversification de grilles et l'etude de partage des gains, pas la promesse de predire
le prochain tirage.
