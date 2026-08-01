# Analyse de faisabilite

## 1. Question testee

Il faut separer trois objectifs souvent confondus :

1. **Predire le tirage** : attribuer aux numeros des probabilites meilleures que l'uniforme.
2. **Gagner souvent** : augmenter la frequence d'au moins un petit rang avec plusieurs grilles.
3. **Etre rentable** : obtenir un retour total superieur aux mises.

Acheter davantage de combinaisons peut ameliorer le point 2, mais augmente le cout dans la meme
proportion et ne resout pas le point 3. Un systeme multiple est seulement un lot de grilles simples.

## 2. Limite mathematique

Le tirage principal contient cinq numeros parmi 49 et un numero Chance parmi 10 :

```text
C(49, 5) x 10 = 1 906 884 x 10 = 19 068 840 issues
```

Si les tirages sont independants et uniformes, pour toute combinaison `g` et tout historique `H` :

```text
P(prochain tirage = g | H) = 1 / 19 068 840
```

Un reseau neuronal, une regression, les numeros chauds, les retards et les cycles ne peuvent pas
extraire une information absente. Ils peuvent seulement ajuster le bruit de l'echantillon.

## 3. Frequence de gain contre rendement

Le reglement donne environ une chance sur 5,99 d'atteindre un rang. Cette frequence inclut le
rang 9, fixe a `2,20 EUR`, qui rembourse la grille sans benefice. Depuis le 4 mai 2026, la part
globale des mises devolue aux gagnants du Loto est `54,35 %`.

Pour `M` euros mises, le calcul structurel est donc :

```text
retour moyen global = 0,5435 x M
perte moyenne globale = 0,4565 x M
```

Une grille a chaque tirage, trois fois par semaine, represente environ `343,20 EUR` par an et une
perte moyenne globale de `156,67 EUR`. La variance est enorme : le resultat observe peut s'en
ecarter fortement, surtout a court terme.

## 4. Angles qui restent testables

### Biais physique ou operationnel

Un biais de machine, de jeu de boules ou de procedure est possible en theorie. Il doit etre assez
grand pour etre detecte malgre le faible nombre de tirages, rester stable apres un changement de
materiel et continuer sur des donnees futures. Un resultat significatif sur les donnees ayant servi
a choisir l'hypothese ne suffit pas.

### Reports de jackpot

Un report apporte au tirage courant de l'argent finance auparavant. L'esperance d'un tirage peut
donc etre moins defavorable que la moyenne globale. Un calcul naif exige deja un jackpot solo de
`2,20 x 19 068 840 = 41 951 448 EUR` pour que le jackpot seul rembourse statistiquement une
grille. Il faut ensuite modeliser les autres rangs, le nombre de participants et le partage entre
co-gagnants. Les grosses cagnottes attirent justement davantage de grilles.

### Choix moins populaires

Toutes les combinaisons sortent avec la meme probabilite, mais elles ne sont probablement pas
jouees avec la meme frequence. Eviter les dates de naissance (numeros limites a 31), suites et
motifs visuels peut reduire le nombre de co-gagnants conditionnellement a un jackpot. Sans donnees
de mises par combinaison, cela reste une heuristique non calibree.

### Couverture et diversification

Avec un budget fixe de plusieurs grilles, eviter les doublons et limiter leur recouvrement augmente
la diversite des issues couvertes. Cela reduit certaines correlations et modifie la variance, mais
pas l'esperance par euro.

## 5. Protocole scientifique

Le projet applique le protocole suivant :

1. Charger et valider les archives officielles.
2. Normaliser les archives dans SQLite et separer strictement les regimes `6/49` et `5/49`.
3. Tester uniformite, paires et dependances avec simulations Monte-Carlo.
4. Definir les modeles avant d'observer la periode de test.
5. Effectuer un backtest walk-forward : seul le passe est accessible a chaque prediction.
6. Comparer le score de Brier au modele uniforme `5/49`.
7. Penaliser les essais multiples et confirmer sur une nouvelle periode verrouillee.
8. Convertir l'eventuel gain predictif en esperance monetaire nette avant toute mise.

## 6. Criteres de decision

Un modele ne merite une experimentation monetaire que s'il remplit simultanement ces conditions :

- amelioration hors echantillon face a l'uniforme ;
- intervalle d'incertitude excluant zero apres correction des essais multiples ;
- stabilite sur plusieurs fenetres temporelles et regimes de tirage ;
- avantage assez grand pour compenser `45,65 %` de perte structurelle, les partages et les erreurs
  de modele ;
- regle de mise fixee avant les resultats et budget entierement perdable.

En pratique, le quatrieme seuil est extraordinairement eleve. Le resultat le plus probable du
projet est un verdict negatif utile : aucun signal reproductible et donc aucune mise recommandee.

## 7. Stockage reproductible

Les ZIP officiels sont conserves localement dans `data/`. La commande `build-db` construit une
base SQLite normalisee avec cinq tables :

- `sources` : provenance, empreinte SHA-256 et date d'import ;
- `draws` : jeu, regime, date, numero Chance ou complementaire ;
- `draw_numbers` : numeros principaux par position ;
- `prizes` : nombre de gagnants et rapport de chaque rang lorsqu'ils existent dans l'archive.
- `code_prizes` : nombre de codes gagnants et rapport unitaire lorsqu'ils existent.

La contrainte d'unicite rend l'import idempotent. La base n'est pas publiee dans Git : elle est
reconstruite depuis les archives FDJ afin de rester actualisable et verifiable.

## 8. Couche ML

Le modele ML ne recoit jamais le resultat a predire dans ses variables. Pour chaque date, les
variables sont construites avant que tous les tirages de cette date soient ajoutes a l'etat :

- frequences cumulatives et fenetres de 10, 50 et 200 tirages ;
- temps ecoule depuis la derniere apparition ;
- presence lors de la date precedente ;
- affinite historique avec les numeros de la date precedente ;
- jour de semaine, annee et type Loto/Super Loto/Grand Loto ;
- effet propre a chacun des 49 numeros.

Trois familles sont comparees : posterior bayesien regularise, regression logistique penalisee et
gradient boosting avec regularisation L2. La selection des hyperparametres et du poids de
retraction vers l'uniforme est effectuee dans une fenetre temporelle interne; seul le fold externe
sert a annoncer la performance.

Les criteres de qualification sont cumulatifs : delta Brier moyen negatif, borne haute bootstrap
a 95 % negative et test de permutation corrige par Holm inferieur a 5 %. Sans cela, l'API renvoie
`abstention`. Cette regle empeche de transformer le meilleur modele d'un groupe de modeles tous
mauvais en faux pronostic.

## 9. Valeur monetaire

Le moteur de valeur utilise les tirages possedant les neuf rangs modernes. Le rang 9 estime la
participation car son esperance de gagnants est le nombre de grilles multiplie par sa probabilite
exacte. Une validation temporelle compare Ridge et gradient boosting a une mediane par jeu et jour.
La retransformation vers un nombre de grilles applique un facteur de smearing calcule sur une
validation passee et publie son biais en niveau hors echantillon.

Le volume retenu alimente un partage Poisson du jackpot et rapporte le pool moyen des codes au
nombre de grilles. Les petits rangs sont agreges comme moyenne des esperances completes par tirage,
ce qui evite le biais d'une mediane calculee separement pour chaque rang. Un bootstrap predictif
echantillonne un prochain bareme historique et l'erreur de participation. Les seuils central et
conservateur reajustent le volume pour chaque jackpot candidat et signalent un depassement du
support d'apprentissage. La decision reste `no_bet` tant que sa borne basse ne depasse pas le prix.
La popularite reelle d'une combinaison reste inconnue et se teste seulement par scenario.

## 10. Backtest de valeur

Une estimation monetaire datee exclut tous les tirages egaux ou posterieurs a sa cible. Le backtest
decoupe ensuite les dates futures en folds bloques. Au debut de chaque fold, il reajuste le modele
de participation, choisit une fenetre predictive de baremes sur une validation interne, puis gele
ces choix pour toute la periode test.

La cible n'est pas le gain aleatoire d'une grille particuliere. Elle reconstruit l'esperance du
bareme futur avec les probabilites exactes, le volume deduit du rang 9, les codes publies et un
partage Poisson moyen. Le protocole annonce biais, MAE, RMSE, couverture, intervalle de Wilson,
delta apparie face a une reference sans partage ni codes et nombres de faux positifs/negatifs.
