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

### Validation ML

Executer les modeles bayesien, logistique et gradient boosting :

```bash
loto-lab ml-backtest data/loto.sqlite \
  --min-history 50 --min-train 500 --folds 3 --simulations 2000 \
  --output reports/ml-backtest.json
```

Le protocole utilise une validation temporelle imbriquee : chaque fold externe mesure la
generalisation et une fenetre interne choisit les hyperparametres. Les variables incluent les
frequences sur 10/50/200 tirages, le retard, les paires avec la date precedente, le jour, le type
de jeu, la tendance temporelle et un effet regulier par numero. Les probabilites sont reprojetees
pour que leur somme soit exactement 5. Une intensite choisie dans la validation interne retracte
ensuite chaque modele vers l'uniforme; une intensite nulle annule tout signal non reproductible.

Demander une prediction avec abstention obligatoire :

```bash
loto-lab ml-predict data/loto.sqlite --date 2026-08-01 --game loto
```

Le programme ne renvoie des numeros que si la borne haute a 95 % du delta Brier est negative et
si le test de permutation reste significatif apres correction de Holm. `--force` existe pour les
experiences, mais sa sortie porte explicitement le statut `forced_experimental`.

### Esperance monetaire

Valider l'estimation du nombre de grilles jouees :

```bash
loto-lab participation-backtest data/loto.sqlite --min-train 500 --folds 3
```

Le volume est estime par `gagnants du rang 9 / probabilite du rang 9`, puis une regression et un
gradient boosting sont compares a une mediane historique par jeu et jour. Seul un modele dont
l'intervalle apparie et la permutation corrigee battent cette reference est retenu. La
retransformation logarithmique applique un facteur de smearing estime dans le passe de chaque fold;
la sortie publie egalement le biais en niveau avant et apres calibration.

Evaluer automatiquement un jackpot annonce :

```bash
loto-lab value data/loto.sqlite --game loto --jackpot 10000000 --date 2026-08-01
```

La decision vaut `eligible` seulement si la borne basse bootstrap de l'esperance depasse le prix
de la grille. Le calcul estime la participation, le partage Poisson du jackpot et la valeur des
codes historiques. `--popularity-factor` teste la popularite relative d'une combinaison;
`--co-winners` remplace explicitement le modele pour un scenario manuel. Le rapport distingue le
jackpot d'equilibre central du jackpot ou la borne basse atteint le prix, recalcule la participation
a chaque niveau et signale toute extrapolation au-dela des jackpots observes.

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
