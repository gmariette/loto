# Loto Lab

Laboratoire Python reproductible pour tester si l'historique Loto, Super Loto et Grand Loto
de la FDJ contient un signal
predictif exploitable. Le projet calcule les probabilites exactes, audite l'uniformite des
tirages, compare plusieurs heuristiques au hasard par backtest chronologique, simule une
bankroll, execute une validation ML imbriquee, estime la participation et genere des grilles
diversifiees.

## Conclusion honnete

Dans un tirage independant et uniforme, aucun algorithme ne peut predire la prochaine
combinaison mieux que le hasard a partir des tirages precedents. Au 1er aout 2026 :

- une grille coute `2,20 EUR` ;
- le jackpot a une probabilite de `1 / 19 068 840` ;
- un gain, remboursement inclus, arrive environ `1 / 5,985` ;
- le taux global des mises devolu aux gagnants est `54,35 %` depuis le 4 mai 2026 ;
- l'esperance de perte structurelle est donc proche de `45,65 %` des mises sur le long terme.

Le taux de gain de `16,7 %` ne signifie pas rentabilite. Le rang 9 rembourse seulement la mise.
Cent grilles coutent `220 EUR` et ont un retour moyen global theorique de `119,57 EUR`, soit une
perte moyenne de `100,43 EUR`. Les reports et gains variables changent un tirage particulier,
pas cette conclusion globale.

## Ce que le projet peut vraiment chercher

1. Detecter un ecart statistique anormal dans des boules, paires ou transitions.
2. Verifier si cet ecart survit sur une periode future jamais utilisee pour regler le modele.
3. Mesurer l'avantage avec le score de Brier face au modele uniforme.
4. Diversifier plusieurs grilles afin d'eviter les doublons et la correlation inutile.
5. Eviter des choix humains populaires pour reduire, sans garantie, le risque de partager un
   jackpot si la combinaison gagne.

Une frequence historique elevee n'est pas une preuve de causalite. Le generateur `anti-crowd`
n'augmente pas la probabilite de sortie des numeros.

## Installation

Python 3.11 ou plus recent :

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

Le coeur statistique utilise NumPy et scikit-learn. Le format XLSX est optionnel :

```bash
python -m pip install -e '.[xlsx]'
```

## Utilisation

Calculer les chances et l'esperance de 100 grilles :

```bash
loto-lab odds --grids 100
```

Telecharger les 11 archives officielles publiees par la FDJ :

```bash
loto-lab download-all --output-dir data
loto-lab build-db --archives-dir data --db data/loto.sqlite --from-year 1996
loto-lab db-info --db data/loto.sqlite
```

Si la chaine TLS de Python est defectueuse sur macOS, le telechargement retente avec `curl` en
conservant la verification des certificats; aucun mode non securise n'est utilise.

La base SQLite locale contient `5 566` tirages entre 1996 et juillet 2026, leurs numeros, le jeu,
le regime, l'archive source, `40 660` observations par rang et `1 519` gains par code. Elle est
reproductible et ignoree par Git afin d'eviter de versionner des donnees derivees et vite obsoletes.

Analyser le regime actuel `5/49 + Chance` directement depuis SQLite :

```bash
loto-lab analyze data/loto.sqlite --simulations 5000
```

Analyser separement le regime historique `6/49 + complementaire` depuis 1996 :

```bash
loto-lab analyze-legacy data/loto.sqlite --from-year 1996 --simulations 5000
```

Tester les modeles uniquement dans l'ordre du temps :

```bash
loto-lab backtest data/loto.sqlite --min-train 500
```

Une valeur `mean_delta` negative indique un score de Brier meilleur que l'uniforme. Elle ne
devient interessante que si elle est stable sur une periode future et statistiquement nette apres
correction des essais multiples.

Toutes les API datees appliquent une coupure stricte: un tirage dont la date est egale ou
posterieure a la cible est exclu de l'apprentissage. `ml-predict` refuse une cible deja presente
dans les donnees fournies.

### Validation ML

Executer les modeles bayesien cumulatif, bayesien temporel, logistique probabiliste, logistique
optimisee pour le Top-5, Ridge intra-tirage, deux Ridge sur frequences glissantes et gradient
boosting sur le jeu cible uniquement :

```bash
loto-lab ml-backtest data/loto.sqlite \
  --game loto --min-history 50 --min-train 500 --folds 3 --simulations 2000 \
  --as-of 2026-07-29 \
  --output reports/ml-backtest.json
```

Le protocole utilise une validation temporelle imbriquee : chaque fold externe mesure la
generalisation et une fenetre interne choisit les hyperparametres. Les variables incluent les
frequences sur 10/50/200 tirages, le retard, les paires avec la date precedente, le jour, la
tendance temporelle et un effet regulier par numero. Le challenger bayesien temporel choisit sa
fenetre et son a priori dans le passe. Les probabilites sont reprojetees
pour que leur somme soit exactement 5. Une intensite choisie dans la validation interne retracte
ensuite chaque modele vers l'uniforme; une intensite nulle annule tout signal non reproductible.
Le `logistic_ranker` choisit au contraire sa regularisation sur les hits internes et conserve le
classement complet. Le `ridge_ranker` centre les 49 candidats de chaque tirage avant apprentissage :
seules les differences entre numeros peuvent influencer leur ordre. Le `rolling_ridge_ranker`
elimine les biais fixes et ne conserve que les frequences passees sur 10, 50 et 200 tirages. Chaque
famille possede une graine stable independante de son ordre dans la liste. Le
`hierarchical_ridge_ranker` reprend ce score et departage seulement ses ex aequo par le retard
normalise, puis par une graine stable si les deux criteres sont encore identiques. `--as-of` fige
explicitement la derniere date admise dans une experience reproductible.

Demander une prediction avec abstention obligatoire :

```bash
loto-lab ml-predict data/loto.sqlite --date 2026-08-01 --game loto
```

Le programme ne renvoie des numeros que si le modele se qualifie sur l'une de deux voies : Brier
significativement meilleur que l'uniforme, ou gain de hits Top-5 dont la borne basse est positive.
Les p-values des huit modeles et des deux metriques sont corrigees ensemble par Holm. `--force`
existe pour les experiences, mais sa sortie porte explicitement le statut `forced_experimental`.
Elle publie aussi le jeu, la cible, la derniere date d'apprentissage, le delta Brier, son intervalle,
les hits Top-5, leurs intervalles et p-values corrigees, ainsi que l'amplitude des probabilites afin
qu'une grille ne soit jamais detachee de sa preuve de non-qualification.

Figer une experience sans mise avant tirage, puis la noter apres publication du resultat :

```bash
loto-lab ml-record data/loto.sqlite --date 2026-08-01 --game loto \
  --seed 20260801 --force --export evidence/number-prospective-ledger.json
loto-lab ml-score data/loto.sqlite \
  --result-source https://www.fdj.fr/jeux-de-tirage/loto/resultats/... \
  --export evidence/number-prospective-ledger.json
loto-lab ml-ledger-verify evidence/number-prospective-ledger.json
```

Le registre des numeros est append-only et separe les changements scientifiques par empreinte du
code, des parametres et des dependances. Chaque cohorte est jugee sur ses 100 premiers scores. Pour
empecher des versions successives de multiplier les chances de faux positif, la cohorte `i` recoit
le budget `0,05 / (i * (i + 1))`; la somme de tous les essais presents et futurs reste sous 5 %.
Le test prospectif utilise la loi hypergeometrique exacte des hits Top-5. Une qualification
historique seule ne suffit donc pas a revendiquer un avantage exploitable.

### Popularite et partage

Valider si la structure d'une combinaison explique le nombre de gagnants du jackpot, apres
correction du volume de tickets estime par le rang 9 :

```bash
loto-lab popularity-backtest data/loto.sqlite \
  --game loto --min-train 500 --folds 3 --simulations 2000 --block-size 12
```

Le modele de Poisson utilise le volume comme exposition et compare sa deviance a une popularite
uniforme calibree dans chaque fold. Ses variables structurelles sont limitees aux effets dont le
signe reste stable dans les validations temporelles : nombres au-dessus de 31, grille entierement
dans 1-31, paires consecutives, 7/13, somme et distance a la somme centrale. L'intervalle et la
p-value utilisent des blocs temporels pour conserver l'autocorrelation des habitudes de jeu.

Une prediction experimentale peut ensuite chercher exhaustivement la combinaison la moins
populaire parmi celles qui respectent un budget explicite de perte de hits attendus :

```bash
loto-lab ml-predict data/loto.sqlite --date 2026-08-03 --game loto \
  --seed 20260803 --force --value-aware --max-expected-hit-loss 0.005 \
  --popularity-bootstrap-models 100 --popularity-uncertainty-quantile 0.9
```

La sortie publie la grille Top-5 sans penalite, la grille retenue, les deux scores de hits attendus,
la perte consentie, le multiplicateur ponctuel, sa borne conservatrice et le backtest complet du
modele de foule. La borne est le quantile demande de modeles reajustes sur des blocs temporels;
l'optimiseur minimise cette borne plutot que l'estimation ponctuelle. Cette optimisation ne change
pas la probabilite du tirage et ne rend pas l'esperance globale positive; elle vise uniquement a
reduire un partage eventuel des gros rangs. La qualification est historique et doit encore etre
confirmee sur des tirages futurs pre-enregistres.

Figer les coefficients avant un tirage, les noter apres publication des rapports puis verifier la
preuve autonome :

```bash
loto-lab popularity-record data/loto.sqlite --date 2026-08-03 --game loto \
  --seed 20260803 --ledger data/popularity-prospective.sqlite \
  --export evidence/popularity-prospective-ledger.json
loto-lab popularity-score data/loto.sqlite \
  --result-source https://www.fdj.fr/jeux-de-tirage/loto/resultats/... \
  --export evidence/popularity-prospective-ledger.json
loto-lab popularity-ledger-verify evidence/popularity-prospective-ledger.json
```

Le snapshot contient les coefficients bruts, les variables, la reference uniforme, le backtest,
les empreintes du code, des dependances et des donnees. Chaque score recalcule le volume depuis le
rang 9 et compare les deviances de Poisson sur la combinaison reellement sortie. Seuls les 100
premiers scores d'une cohorte comptent. La cohorte `i` utilise le budget alpha
`0,05 / (i * (i + 1))`; une borne haute du delta strictement negative et une permutation par blocs
sous ce budget sont necessaires. Avant ces 100 observations, le statut reste
`insufficient_data` et aucun avantage prospectif n'est revendique.

### Esperance monetaire

Valider l'estimation du nombre de grilles jouees :

```bash
loto-lab participation-backtest data/loto.sqlite --min-train 500 --folds 3
```

Le volume est estime par `gagnants du rang 9 / probabilite du rang 9`, puis une regression et un
gradient boosting sont compares a une mediane historique par jeu et jour. Le jackpot annonce est
reconstruit comme le pool total lorsque plusieurs gagnants se le partagent. Seul un modele dont
l'intervalle apparie et la permutation corrigee battent la reference est retenu. La retransformation
logarithmique applique un facteur de smearing estime dans le passe de chaque fold; l'incertitude
utilise ensuite les erreurs de niveau reellement observees hors echantillon, normalisees a moyenne 1.

Evaluer automatiquement un jackpot annonce :

```bash
loto-lab value data/loto.sqlite --game loto --jackpot 10000000 --date 2026-08-01
```

La decision vaut `eligible` seulement si la borne basse bootstrap de l'esperance depasse le prix
de la grille. L'horizon historique et la probabilite de queue sont choisis dans une validation
temporelle passee pour viser 95 % de couverture empirique. Le calcul estime la participation, le
partage Poisson du jackpot et la valeur des codes historiques. `--popularity-factor` teste la
popularite relative d'une combinaison;
`--co-winners` remplace explicitement le modele pour un scenario manuel. Le rapport distingue le
jackpot d'equilibre central du jackpot ou la borne basse atteint le prix, recalcule la participation
a chaque niveau et signale toute extrapolation au-dela des jackpots observes. Depuis la version
0.8.0, `expected_return_rate` designe `EV / prix` et `estimated_roi` designe le rendement net
`(EV - prix) / prix`; un taux de retour de 49 % correspond donc a un ROI net de -51 %. Le champ
`naive_ev` fige aussi la reference sans partage, codes ni modele de participation qui servira a
l'evaluation prospective comparative.

Backtester le moteur monetaire complet sur des folds futurs :

```bash
loto-lab value-backtest data/loto.sqlite --game loto \
  --min-train 500 --folds 3 --refit-interval 52 \
  --simulations 2000 --block-size 12 --seed 42
```

La cible est l'esperance reconstruite depuis le bareme futur publie, les codes et le volume estime
par le rang 9. La sortie compare le modele a un calcul naif sans partage ni codes, teste le delta
MAE par bootstrap et permutation en blocs temporels, mesure la couverture predictive avec le meme
re-echantillonnage chronologique et compte les faux `eligible`. Dans chaque fold externe, le modele
est reentraine uniquement sur le passe toutes les 52 dates; chaque periode publie son propre biais,
sa MAE, sa couverture et ses erreurs de decision.

### Validation prospective

Figer une unique estimation avant un tirage et publier sa preuve :

```bash
loto-lab value-record data/loto.sqlite \
  --ledger data/prospective.sqlite --game loto --jackpot 5000000 \
  --date 2026-08-01 \
  --jackpot-source https://www.fdj.fr/jeux-de-tirage/loto/resultats/mercredi-29-juillet-2026 \
  --simulations 2000 --seed 42 --export evidence/value-2026-08-01.json
loto-lab ledger-info --ledger data/prospective.sqlite
```

Le registre SQLite refuse la retroactivite, les doublons jeu/date, les mises a jour et les
suppressions. Chaque prevision et chaque score prolongent une chaine SHA-256. Publier l'export JSON
et le hash de tete dans Git avant le tirage fournit l'ancrage temporel externe; la base locale seule
ne suffit pas, car son historique complet pourrait etre recree. Depuis la version 0.9.0, la preuve
contient aussi la taille et le SHA-256 de chaque fichier d'entree, plus une empreinte logique de tous
les tirages effectivement charges.

Apres actualisation des archives et de la base des tirages :

```bash
loto-lab value-score data/loto.sqlite --ledger data/prospective.sqlite \
  --result-source https://www.fdj.fr/jeux-de-tirage/loto/resultats/... \
  --export evidence/prospective-ledger.json
loto-lab ledger-verify evidence/prospective-ledger.json
```

Le score utilise le bareme, les codes et le volume deduit du rang 9. `ledger-info` ne calcule biais,
MAE, couverture et erreurs de decision que sur ces previsions figees. Le bundle autonome contient
les previsions, les tirages, les scores et les deux chaines; `ledger-verify` recalcule aussi chaque
metrique depuis le bareme sans acceder a SQLite. Un score est refuse avant 20 h 15, heure de Paris,
le jour cible. La source du resultat doit etre une URL HTTPS du domaine FDJ et fait partie du hash
du score. Le premier ancrage public est documente dans [PROSPECTIVE.md](PROSPECTIVE.md).

`ledger-info` regroupe les scores comparables par `evaluation_cohort`, l'empreinte SHA-256 de la
specification scientifique publiee avant tirage. Elle couvre les sources du moteur de valeur, ses
parametres effectifs, Python, NumPy et scikit-learn. Une mise a jour de documentation ou du cycle
operationnel ne remet donc plus le compteur a zero; une modification scientifique ouvre
automatiquement une nouvelle cohorte. Aucun verdict n'est produit avant 100 scores par empreinte.
A ce seuil, bootstrap et permutation par blocs de 12 comparent les erreurs absolues au benchmark et
controlent la couverture de 95 %. Le verdict est ensuite fige sur ces 100 observations; les
resultats suivants alimentent uniquement les champs `monitoring_*`.

### Cycle operationnel

Preparer un manifeste depuis [l'exemple](examples/prospective-manifest.example.json), puis verifier
le plan sans aucune ecriture :

```bash
loto-lab prospective-run examples/prospective-manifest.example.json --dry-run
```

Avant le tirage, renseigner date, jackpot et `jackpot_source`, laisser `result_source` a `null`, puis
passer explicitement `enabled` a `true` et executer sans `--dry-run`. L'exemple desactive refuse toute
execution reelle. Apres publication du resultat, actualiser les archives et la base, renseigner
`result_source`, puis relancer exactement la meme commande. Le cycle detecte les etats `missing`,
`pending` et `scored`; il ne cree jamais deux previsions ni deux scores pour une cible.

L'export du registre utilise un fichier temporaire et un remplacement atomique. Si l'execution est
interrompue apres la transaction SQLite mais avant l'export, le passage suivant reconstruit la preuve
sans dupliquer la ligne. Chaque passage reel prolonge aussi un journal JSONL chaine :

```bash
loto-lab prospective-journal-verify data/prospective-operations.jsonl
```

Ce journal sert a l'audit operationnel local. Comme toute chaine locale, il ne prouve sa chronologie
que si sa tete est publiee dans Git ou un autre support externe.

Les chemins relatifs du manifeste sont resolus depuis son propre dossier. Il peut donc etre lance
depuis cron ou un runner CI sans dependre du repertoire courant.

Generer dix grilles distinctes :

```bash
loto-lab generate --count 10 --mode anti-crowd --seed 42
```

Simuler une bankroll avec des montants de gains explicitement fournis :

```bash
loto-lab simulate --payouts examples/payouts.example.json --tickets 2 --draws 156 --runs 10000
```

Le fichier d'exemple contient des montants illustratifs. Les rangs 1 a 8 sont variables et doivent
etre remplaces par les montants du tirage etudie. La simulation ne modelise pas les co-gagnants.

## Donnees et formats

`loto-lab` lit les ZIP et CSV officiels, SQLite, ainsi que les XLSX avec l'option correspondante. Les
colonnes attendues sont `date_de_tirage`, les colonnes `boule_N` et le numero special. Les doublons
sont retires et les tirages dates sont tries chronologiquement. Les archives sont segmentees ainsi :

- `2 859` tirages compatibles `5/49 + Chance` depuis octobre 2008 : 2 788 Loto, 60 Super Loto
  et 11 Grand Loto ;
- `2 707` tirages `6/49 + complementaire` entre janvier 1996 et octobre 2008 : 2 664 Loto et
  43 Super Loto.

Ces regimes ne sont jamais fusionnes dans un test ou un backtest.

Sources officielles consultees :

- [Reglement homologue par l'ANJ en avril 2026](https://www.anj.fr/sites/default/files/2026-04/D%C3%A9cision%20n%C2%B02026-PR-034_HOM%20R%C3%A8glement_Loto2dTirage_PDV-Ligne_LFDJ.pdf)
- [Historique telechargeable FDJ](https://www.fdj.fr/jeux-de-tirage/loto/historique)
- [Statistiques et probabilites FDJ](https://www.fdj.fr/jeux-de-tirage/loto/statistiques)

L'[analyse de faisabilite](ANALYSE.md) decrit les hypotheses et les seuils de decision. Les
[resultats reproductibles](RESULTATS.md) donnent le verdict obtenu sur l'ensemble des archives.
La [model card](MODEL_CARD.md) documente les variables, usages autorises et limites du ML.

## Jeu responsable

Ce logiciel est un outil de recherche, pas un conseil financier et pas un systeme de jeu
rentable. Fixer une mise maximale entierement perdable reste la seule contrainte de bankroll
robuste. En France, Joueurs Info Service est joignable au `09 74 75 13 13`.
