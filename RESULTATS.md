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

Un delta positif est moins bon. Les trois modeles historiques etaient donc significativement moins
bien calibres que la probabilite uniforme `5/49`. Le leger surplus de Top-5 du premier modele ne
compense pas ses probabilites moins bien calibrees et n'implique aucun avantage monetaire.

## Validation ML imbriquee

La version 0.13 ajoute un posterior bayesien temporel dont la fenetre 10/50/200 et la force de
l'a priori sont choisies dans chaque fenetre interne. Elle isole aussi le jeu cible au lieu de
melanger Loto, Super Loto et Grand Loto. Trois folds externes, 2 000 bootstraps et permutations,
puis une correction de Holm entre modeles evaluent 2 238 tirages Loto jamais utilises pour ajuster
le fold correspondant.

| Modele | Delta Brier | IC 95 % du delta | Log-loss | Hits Top-5 | Qualifie |
|---|---:|---:|---:|---:|---:|
| Bayesien cumulatif | +0,00000478 | [-0,00000063 ; +0,00001026] | 0,32957 | 0,5152 | non |
| Bayesien temporel | +0,00000143 | [-0,00000190 ; +0,00000485] | 0,32955 | 0,4933 | non |
| Logistique retractee | +0,00000321 | [-0,00000675 ; +0,00001362] | 0,32956 | 0,5232 | non |
| Gradient boosting retracte | 0 | [0 ; 0] | 0,32954 | 0,5147 | non |

La log-loss uniforme vaut `0,32954`. Le challenger temporel reduit d'environ 70 % la degradation du
bayesien cumulatif, mais ne franchit pas zero. La logistique obtient davantage de hits Top-5 sans
ameliorer significativement son score probabiliste. Le gradient boosting choisit une retraction
totale et reproduit exactement le benchmark. Aucun modele ne se qualifie; `ml-predict` renvoie donc
`abstention`.

La version 0.14 teste aussi directement le classement d'une grille. Les egalites de probabilites
sont departagees par un alea reproductible, puis les hits sont compares a `25/49`. L'intervalle est
bootstrappe et la p-value vient du null hypergeometrique exact; Holm corrige conjointement les quatre
tests Brier et les quatre tests Top-5.

| Modele | Hits Top-5 | Gain vs 25/49 | IC 95 % du gain | p Holm | Qualifie classement |
|---|---:|---:|---:|---:|---:|
| Bayesien cumulatif | 0,5152 | +0,0050 | [-0,0232 ; +0,0318] | 1,000 | non |
| Bayesien temporel | 0,4933 | -0,0169 | [-0,0437 ; +0,0099] | 1,000 | non |
| Logistique retractee | 0,5232 | +0,0130 | [-0,0133 ; +0,0398] | 1,000 | non |
| Gradient boosting retracte | 0,5147 | +0,0045 | [-0,0209 ; +0,0309] | 1,000 | non |

La logistique reste la meilleure en hits bruts, mais son intervalle couvre largement zero. Cette
nouvelle metrique ne justifie donc toujours pas une grille predictive.

## Participation et esperance monetaire

Le nombre de gagnants du rang 9 permet d'estimer le nombre de grilles sur 1 519 tirages modernes.
La version 0.5.0 reconstruit le jackpot annonce comme le pool total lors des 87 tirages modernes a
plusieurs gagnants. Sur 1 019 observations futures, le gradient boosting atteint une RMSE
logarithmique de `0,2154` contre `0,2315` pour la mediane segmentee, soit `6,97 %` d'amelioration.
Le delta MSE apparie a un IC 95 % de `[-0,01114 ; -0,00324]` et une p-value corrigee de Holm de
`0,003`; il est qualifie. Ridge progresse de `3,10 %` et se qualifie aussi, mais reste moins precis.

La retransformation calibree du logarithme porte le biais agrege en niveau du gradient boosting de
`-4,65 %` brut a `-3,32 %`; les facteurs sont calcules uniquement dans le passe de chaque fold.
L'incertitude n'impose plus une loi normale: elle re-echantillonne 1 019 erreurs multiplicatives
hors echantillon, normalisees a moyenne 1.

Pour un Loto a 10 M EUR le 1er aout 2026, le modele estime `5,38 millions` de grilles, `0,282`
co-gagnant moyen et un facteur de partage du jackpot de `0,871`. Les petits rangs sont maintenant
agreges au niveau tirage par une moyenne, methode adaptee a une esperance, au lieu de medianes rang
par rang. Avec environ `0,037 EUR` de codes, l'esperance vaut `1,305 EUR` pour `2,20 EUR`, soit un
taux de retour de `59,32 %` et un ROI net de `-40,68 %`.

La version 0.6.0 selectionne conjointement l'horizon de baremes et une probabilite de queue dans
une validation temporelle. Pour ce rapport, 1 176 tirages et une queue de 1 % donnent `95,59 %` de
couverture passee. L'intervalle predictif calibre vaut `[1,038 ; 1,639]`. Le seuil central reste
`30,227 M EUR`; la borne basse atteint le prix vers `36,925 M EUR`. Les deux depassent le jackpot
Loto maximal de `30 M EUR` observe dans les archives et sont marques comme extrapolations. La
decision reste `no_bet`.

## Backtest bout en bout de la valeur

La version 0.6.0 transforme les trois folds externes en walk-forward periodique. Le moteur est
reentraine toutes les 52 dates et produit 21 periodes sur 983 tirages Loto du 20 avril 2020 au
29 juillet 2026. Chaque ajustement n'utilise que les tirages strictement anterieurs a sa periode.
La cible est l'esperance du bareme observe, reconstruite avec les petits rangs, les codes et le
volume du rang 9.

| Mesure | Modele complet | Reference naive |
|---|---:|---:|
| Biais moyen | +0,00169 EUR | -0,00829 EUR |
| MAE | 0,11157 EUR | 0,11717 EUR |
| RMSE | 0,13752 EUR | 0,14686 EUR |

La MAE baisse de `4,78 %`. Le delta apparie vaut `-0,00560 EUR`. Un bootstrap en blocs contigus de
12 tirages donne l'IC 95 % `[-0,00808 ; -0,00337]`; la permutation par blocs donne `p = 0,0005`.
L'amelioration face a la reference naive reste donc qualifiee sans supposer les erreurs voisines
independantes. La couverture predictive vaut `93,90 %`, avec IC par blocs
`[91,86 % ; 95,83 %]`: la cible de 95 % reste compatible. Le verdict cumulatif exige ces deux
conditions et vaut `value_model_qualified = true`.

Le modele bat la reference dans 19 des 21 periodes. Il fait moins bien de novembre 2024 a fevrier
2025 (`-6,39 %`) et de fevrier a mai 2026 (`-7,33 %`), ce qui interdit de presenter l'avantage
moyen comme permanent. Une sensibilite exploratoire du rythme donne :

| Intervalle de reentrainement | Refits | Biais | MAE |
|---|---:|---:|---:|
| Fige par fold, v0.5.0 | 3 | +0,00475 EUR | 0,11160 EUR |
| 52 dates, protocole principal | 21 | +0,00169 EUR | 0,11157 EUR |
| 104 dates, exploratoire | 12 | +0,00123 EUR | 0,11164 EUR |
| 156 dates, exploratoire | 9 | +0,00236 EUR | 0,11166 EUR |

Les variantes 104/156 ont ete examinees sur le test externe et ne servent donc pas a choisir le
protocole principal. Le rythme de 52 dates avait ete fixe avant cette comparaison.

Le moteur n'a emis aucun `eligible` et donc aucun faux positif. Trois baremes observes avaient une
EV ponctuelle superieure au prix, tous refuses par la borne basse prudente. Ce resultat confirme
une politique d'abstention conservatrice; il ne demontre pas une strategie de jeu rentable.

## Validation prospective

La version 0.7.0 arrete l'optimisation repetitive du meme test externe et ouvre un registre
append-only. La premiere prevision a ete creee le 1er aout 2026 a 12:56 UTC, avant le tirage Loto
du jour et avec le jackpot officiel de 5 M EUR. Elle utilise les donnees arretees au 29 juillet :

| Mesure figee | Valeur |
|---|---:|
| EV | 1,08410 EUR |
| Intervalle predictif | [0,81633 ; 1,41460] EUR |
| Taux de retour | 49,28 % |
| ROI net | -50,72 % |
| Decision | `no_bet` |

Le hash public est
`39d08472180461bb0413a405fb705bf8e0bd8cc01522a4477132c870432abbc7`. La preuve complete est
dans `evidence/value-2026-08-01.json`. Aucun score prospectif n'est encore disponible: annoncer une
MAE ou une couverture prospective avec zero resultat serait trompeur.

La version 0.8.0 corrige l'ambiguite de vocabulaire du rapport: le champ v0.7
`estimated_roi = EV / prix` etait en realite un taux de retour. Les nouveaux rapports publient
`expected_return_rate = EV / prix` et `estimated_roi = (EV - prix) / prix`. La preuve v0.7 reste
intacte et cette correction est explicite plutot que retroactive. Un bundle exportable permet en
outre de verifier hors SQLite les deux chaines et de recalculer chaque score depuis son bareme.

La version 0.9.0 rend la provenance reproductible pour toutes les nouvelles lignes. Elle hache les
fichiers d'entree et le snapshot logique des tirages charges. Chaque nouveau score utilise un hash
v2 qui lie aussi l'URL HTTPS FDJ du resultat et ces empreintes. Le registre migre automatiquement
les anciennes lignes en les laissant en hash v1; les preuves v1 et v2 restent verifiables. La
prevision du 1er aout reste volontairement une preuve v0.7 et n'acquiert pas retroactivement une
provenance qu'elle n'avait pas au moment de son ancrage.

La version 0.10.0 ajoute a chaque nouvelle prevision `naive_ev`, calculee exactement comme la
reference du backtest: petits rangs moyens plus valeur faciale du jackpot, sans codes, partage ni
modele de participation. Apres tirage, le score chaine les erreurs absolues du modele et de cette
reference ainsi que leur delta. Dans ce protocole initial, les cohortes etaient separees par version
et aucune qualification prospective n'etait possible avant 100 observations comparables. Le verdict
utilise uniquement les 100 premieres, avec bootstrap et permutation par blocs de 12; les observations
suivantes restent visibles en surveillance mais ne permettent pas de refaire le test jusqu'a obtenir
un succes. La version 0.12.0 remplace ensuite cette cle de regroupement trop large.

Le registre contient actuellement zero score et zero observation avec benchmark. La preuve v0.7
sera exclue de la comparaison au benchmark, car celui-ci n'avait pas ete publie avant son tirage.

La version 0.11.0 automatise le cycle necessaire pour atteindre ces 100 observations. Un manifeste
JSON strict pilote le preflight, l'enregistrement, l'attente du resultat, le scoring et l'export.
Toutes les operations sont idempotentes et chaque execution reelle prolonge un journal JSONL
chaine. Les exports sont atomiques et un passage suivant repare un export absent apres une transaction
SQLite deja validee.

Un dry-run du manifeste d'exemple sur les 2 859 tirages reels a choisi `planned_record`, produit un
rapport complet et laisse registre, preuve et journal inchanges. Le jackpot et la source de cet
exemple sont des valeurs a remplacer; ce dry-run n'est pas une nouvelle prevision officielle.

La version 0.12.0 remplace le regroupement prospectif par version logicielle par une empreinte
scientifique reproductible. Le code du moteur de valeur, les parametres et les dependances
numeriques sont hashes dans chaque nouvelle prevision. Une release sans changement scientifique ne
perd plus les observations accumulees; un changement de modele ne peut pas reutiliser silencieusement
la cohorte precedente. La migration du registre et de la preuve vers le schema 4 ne modifie ni le
payload ni le hash de la prevision v0.7, qui reste une entree historique sans empreinte de modele.

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
