# Resultats sur toutes les archives

Calculs effectues le 1er aout 2026 avec les 11 archives disponibles sur la page historique FDJ.
La base locale contient 5 566 tirages, 40 660 lignes de gains par rang et 1 519 gains par code.

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

La version 0.3 ajoute aux 63 variables une retraction vers l'uniforme choisie dans chaque fenetre
interne. Trois folds externes, 2 000 bootstraps et permutations, puis une correction de Holm entre
modeles evaluent 2 309 tirages jamais utilises pour ajuster le fold correspondant.

| Modele | Delta Brier | IC 95 % du delta | Log-loss | Hits Top-5 | Qualifie |
|---|---:|---:|---:|---:|---:|
| Bayesien retracte | +0,00000498 | [-0,00000022 ; +0,00001013] | 0,32957 | 0,5058 | non |
| Logistique retractee | +0,00001011 | [-0,00000422 ; +0,00002396] | 0,32960 | 0,5154 | non |
| Gradient boosting retracte | +0,00002437 | [+0,00000961 ; +0,00003891] | 0,32968 | 0,5032 | non |

La log-loss uniforme vaut `0,32954`. La retraction reduit la degradation moyenne du bayesien
d'environ 85 % et celle de la logistique d'environ 95 % par rapport a la version 0.2. Elle ne cree
cependant aucun avantage: les deux intervalles couvrent zero et le gradient reste moins bon. Sur
tout l'historique, le poids de production choisi vaut zero pour le bayesien et la logistique.
La commande `ml-predict` renvoie donc `abstention`.

## Participation et esperance monetaire

Le nombre de gagnants du rang 9 permet d'estimer le nombre de grilles sur 1 519 tirages modernes.
Sur 1 019 observations futures, le gradient boosting atteint une RMSE logarithmique de `0,2187`
contre `0,2315` pour la mediane segmentee, soit `5,54 %` d'amelioration. Le delta MSE apparie a un
IC 95 % de `[-0,01011 ; -0,00110]` et une p-value corrigee de Holm de `0,013`; il est qualifie.
La regression Ridge, amelioree de `2,38 %`, ne se qualifie pas car son intervalle couvre zero.

La version 0.3.1 ajoute une retransformation calibree du logarithme. Sur les folds externes, le
biais agrege en niveau du gradient boosting passe de `-4,51 %` a `-3,08 %`; les facteurs sont
calcules uniquement dans le passe de chaque fold. Le rapport donne aussi les bornes de jackpot
observees afin de distinguer interpolation et extrapolation.

Pour un Loto a 10 M EUR le 1er aout 2026, le modele estime `5,41 millions` de grilles, `0,283`
co-gagnant moyen et un facteur de partage du jackpot de `0,871`. Les petits rangs sont maintenant
agreges au niveau tirage par une moyenne, methode adaptee a une esperance, au lieu de medianes rang
par rang. Avec
environ `0,037 EUR` de codes, l'esperance vaut `1,304 EUR` pour `2,20 EUR`, soit un ROI de `59,29 %`.

Le bootstrap predictif echantillonne un prochain bareme complet et l'erreur de participation; son
IC 95 % `[1,072 ; 1,576]` est volontairement plus large que l'ancien intervalle d'estimation. Le
seuil central dynamique vaut `30,281 M EUR`; la borne basse atteint seulement le prix vers
`36,134 M EUR`. Les deux depassent le jackpot Loto maximal de `30 M EUR` observe dans les archives
et sont marques comme extrapolations. La decision reste `no_bet`.

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
