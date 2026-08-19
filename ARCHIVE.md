# Cloture scientifique

Le projet est archive le 19 aout 2026. Son objectif principal etait de verifier si les tirages
precedents permettaient de predire les numeros futurs mieux qu'une reference uniforme. La reponse
mesuree est **non** : aucun avantage predictif reproductible n'a ete demontre.

## Verdict final

La base locale est arretee au tirage Loto du 17 aout 2026. Le dernier backtest imbrique evalue sept
modeles sur 2 246 tirages hors echantillon, avec apprentissage et selection d'hyperparametres
strictement anterieurs a chaque periode de test.

- Le meilleur delta Brier est celui du Ridge glissant : `-0,000000453`, avec un intervalle a 95 %
  de `[-0,000003805 ; +0,000003042]` et une p-value corrigee de `1,000`.
- Le meilleur resultat Top-5 brut est celui du gradient boosting : `0,52861` hit par tirage contre
  `0,51020` pour une grille uniforme. Son uplift a un intervalle de
  `[-0,00393 ; +0,04158]` et une p-value de Holm de `0,51774`.
- Aucun des sept modeles ne satisfait la qualification probabiliste ou la qualification de rang.
- Le replay strict des sept tirages du 3 au 17 aout obtient deux hits contre `3,571` attendus sous
  l'uniforme. Ce resultat est compatible avec le hasard.
- La decision de production reste `abstention`. Les sorties forcees sont experimentales et ne
  doivent pas etre presentees comme des predictions gagnantes.

Les modeles de participation et de partage conservent un interet descriptif, mais aucune strategie
de mise rentable n'a ete demontree. Ajouter de nouvelles transformations des memes tirages ferait
surtout croitre le risque de surapprentissage. Une reprise du projet ne serait justifiee que par de
nouvelles variables exogenes credibles, par exemple des identifiants publics de machine ou de jeux
de boules, suivies du meme protocole prospectif.

## Reproductibilite

Les documents de reference sont :

- `MODEL_CARD.md` pour les modeles, les controles et les limites ;
- `RESULTATS.md` pour les backtests historiques ;
- `PROSPECTIVE.md` pour les predictions figees et leurs scores ;
- `evidence/` pour les registres append-only et les bundles verificables.

Le calcul final peut etre reproduit avec :

```bash
loto-lab ml-backtest data/loto.sqlite \
  --as-of 2026-08-17 --min-history 50 --min-train 500 \
  --folds 3 --simulations 2000 --block-size 12 --seed 0
python3 -m pytest -q
python3 -m ruff check src tests
```

Ce depot est conserve en lecture seule afin de documenter le protocole et le resultat negatif. Il
ne constitue ni un conseil de jeu ni une preuve qu'un avantage futur est impossible si de nouvelles
donnees causales deviennent disponibles.
