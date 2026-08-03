# Registre prospectif

Ce fichier publie les tetes de chaine avant les tirages. La base SQLite locale est append-only et
chaque prevision complete est aussi exportee en JSON suivi par Git. Un hash local non publie ne
prouve pas sa date de creation; le commit distant fournit l'ancrage temporel externe.

## Prevision 1

- creee le `2026-08-01T12:56:01.224781+00:00` ;
- cible : Loto du 1er aout 2026, jackpot annonce de 5 M EUR ;
- dernier tirage d'apprentissage : 29 juillet 2026 ;
- version : `0.7.0`, 2 000 simulations, graine 42 ;
- EV : `1,084099 EUR`, intervalle calibre `[0,816331 ; 1,414598]` ;
- decision : `no_bet` ;
- hash : `39d08472180461bb0413a405fb705bf8e0bd8cc01522a4477132c870432abbc7` ;
- [preuve JSON](evidence/value-2026-08-01.json) ;
- [source officielle du jackpot](https://www.fdj.fr/jeux-de-tirage/loto/resultats/mercredi-29-juillet-2026).

Le score restera absent jusqu'a l'import du bareme officiel du tirage cible. Une seule prevision
par jeu et date est autorisee; elle ne peut pas etre remplacee apres coup.

Note de schema : dans cette preuve v0.7, `estimated_roi` vaut `EV / prix`, donc `49,28 %`. Ce champ
etait mal nomme: il s'agit du taux de retour, tandis que le ROI net vaut `-50,72 %`. La version 0.8
ajoute `expected_return_rate` et reserve `estimated_roi` au ROI net, sans reecrire cette preuve.

Le [snapshot autonome du registre](evidence/prospective-ledger.json) reproduit le meme hash. Il peut
etre controle sans la base locale avec `loto-lab ledger-verify evidence/prospective-ledger.json`.

La version 0.9 ajoute aux nouvelles previsions les hashes des fichiers et du contenu logique charge,
puis lie la source HTTPS FDJ a chaque score. Ces informations ne sont pas ajoutees a cette prevision
v0.7 apres coup: son payload original et son hash public restent la seule preuve recevable.

De meme, la reference `naive_ev` introduite en version 0.10 n'est pas reconstruite apres coup pour
cette prevision. Son futur score contribuera au biais, a la MAE et a la couverture, mais pas au delta
comparatif modele-reference. Seules les previsions qui ont publie les deux EV avant tirage sont
eligibles au protocole de qualification sur 100 observations.

Depuis la version 0.11, les prochaines echeances peuvent etre traitees par `prospective-run` depuis
un manifeste. Le cycle est idempotent, remplace atomiquement ce snapshot et chaine chaque passage
dans `data/prospective-operations.jsonl`. Cette automatisation ne modifie pas la premiere preuve.

Depuis la version 0.12, chaque nouvelle prevision publie aussi son `evaluation_cohort`: le hash du
code scientifique, des parametres et des dependances du moteur de valeur. La prevision v0.7 conserve
une valeur `null`; lui attribuer cette information apres coup creerait une fausse provenance. Si son
score devient comparable, le registre l'isolera dans une cohorte historique liee a sa version.

## Experience de numeros du 1er aout 2026

Une [experience sans mise](evidence/numbers-2026-08-01.json) a ete figee a 16 h 29, heure de Paris,
avant la cloture. Elle compare une sortie ML forcee, trois controles uniformes et trois grilles
anti-partage. Le modele ML n'a produit aucune hierarchie: toutes ses probabilites marginales valent
exactement `5/49`; ses numeros sont donc un repli pseudo-aleatoire reproductible. Cette echeance sert
uniquement a tester le protocole de publication et de scoring, pas a qualifier le modele. Empreinte
du fichier avant tirage : `5063a19ea0879a0c3c4dcb71f4d10d340e4b04932ac76e4fdbe95e8c248edd38`.

La [qualification Top-5 v0.14](evidence/ranking-2026-08-01-v0.14.json), publiee a 17 h 28,
selectionne la logistique comme meilleur challenger brut mais constate une retraction finale
entierement uniforme et aucune qualification. Avec la meme graine, elle conserve exactement la
grille initiale au lieu d'en choisir une seconde apres modification du protocole.
Empreinte : `69910a617dec8029c088d504f2959f222947398928aeb8e83adbc96eaaeca428`.

La [preuve v0.15](evidence/ranking-2026-08-01-v0.15.json), publiee avant la meme cloture,
ajoute une logistique dont la regularisation est choisie directement sur les hits Top-5 internes.
Elle obtient `0,5192` hit hors echantillon contre `0,5232` pour la logistique standard et ne se
qualifie pas. Les graines etant maintenant stables par identite de modele, les resultats des quatre
familles precedentes et la grille `18 35 36 43 46`, Chance `3`, restent inchanges. Empreinte :
`c3ac34a033e7384c8ff542a1e17aac1f19fcfbc261f78246b96c13d9d20dc1e6`.

La version 0.16 ouvre le [registre prospectif des numeros](evidence/number-prospective-ledger.json).
Le Ridge centre dans chaque tirage devient le meilleur challenger historique brut avec `0,5304`
hit Top-5, mais son intervalle couvre zero et il reste `forced_experimental`. La premiere cohorte,
`4be41500c3a70d640cc93763ba4af697875b5e1183a4236a82733e5bc50f91da`, exige 100 scores et
dispose d'un budget alpha de `0,025`; elle en a actuellement zero. La grille sans mise figee a
18 h 55 est `1 11 14 33 43`, Chance `9`. Tete append-only :
`bb1e78b077333218599ea3faadb94083fd09466c5d7566e078d412fb7de87c5b`. Empreinte du bundle :
`d2d45128a316e68fae619a18bc51a90aee57c62cc6c2498fe22de6e4b3f28b3f`.

La version 0.17 ouvre la cohorte 2 sans effacer la premiere. Le Ridge limite aux frequences
10/50/200 atteint `0,5375` hit historique, avec une p-value brute de `0,01999`, mais l'intervalle
touche zero et la p-value Holm vaut `0,27986`. Il reste donc `forced_experimental`. La grille sans
mise figee a 19 h 05 est `6 20 40 42 44`, Chance `9`. La cohorte
`bbcb709cbb00314a2d824ef003af169fb05629f904220df6993642bcc84c7faa` recoit le deuxieme budget
alpha, `0,008333`, et reste a zero score sur 100. Tete append-only :
`87d58a32a7eb11eb105d60990ac0b3a6db8a3553219290f384eecde2ddf57ccf`. Le bundle passe au schema
2 pour conserver la chaine JSON canonique originale et sa representation lisible; le schema 1 reste
verifiable. Empreinte du bundle :
`4ddf625a48a934e4cbc2a204e01494c8afdb87dc882ad755fb98f46fa111fb16`.

Apres publication de l'archive FDJ du 1er aout, le tirage officiel `5 6 8 30 37`, Chance `4`, a
score les deux previsions sans modification retrospective. La v0.16 obtient zero hit principal et
la v0.17 un hit (`6`); aucune ne trouve le numero Chance. Chaque cohorte reste
`insufficient_data` avec un score sur 100. Tete de la chaine des scores :
`c581084c20a25d5af229368b2782b09d2ddd53a07e6d509b8879f01670a5d6cb`. Empreinte du bundle
score : `289068dcd5388665bbda32a4e3b94728b2c2f93754698429944ddee78567cd10`.

La version 0.18 fige la recherche au 29 juillet avec l'option CLI `--as-of`. Le nouveau
`hierarchical_ridge_ranker` conserve le score rolling 10/50/200 et utilise le retard normalise
uniquement pour departager ses ex aequo. Sur les 2 238 tirages externes figes, il atteint `0,53977`
hit contre `0,51020` sous l'uniforme; l'intervalle brut de l'uplift est
`[0,00499 ; 0,05771]` et la p-value brute `0,01649`, mais la correction de Holm sur 16 tests donne
`0,26387`. Il reste donc non qualifie. Les variantes non lineaires, EWMA, multiscales et a fenetre
d'apprentissage recente ont ete rejetees car elles font moins bien hors echantillon.

La grille sans mise figee pour le 3 aout est `11 20 40 42 44`, Chance `3`, entrainee jusqu'au
1er aout. Sur ce snapshot actualise et avec sa graine pre-enregistree, le controle historique vaut
`0,53729` hit et son intervalle recouvre zero; le statut reste `forced_experimental`. La cohorte 3,
`b4dfd1a591d86ffd34a726ea3ef6a98f066a5a88332b2ddc0f18bbc364bc002a`, dispose d'un budget
alpha de `0,0041667`. Tete append-only :
`a2cffc620df43c27749441618ce58207b315f37f6e15ceebbe3456de4fd67c5e`. Empreinte du bundle :
`563e341f2653094d4387797ec2e6e3122ec2899a210d86bbb39d36373825aa59`.

La version 0.19 change l'objectif secondaire plutot que de revendiquer un nouveau signal sur les
boules. Les classements directs, ensembles, walk-forward, multi-origines, interactions par jour et
pooling Loto/Super/Grand Loto ont tous ete rejetes hors echantillon. Le nouveau modele de popularite
utilise en revanche les 1 472 tirages disposant du rang 9 comme exposition et le nombre de gagnants
du rang 1 comme cible de Poisson. Sur 972 observations externes, sa deviance vaut `0,74975` contre
`0,79391` pour la popularite uniforme, delta `-0,04415`, IC temporel par blocs
`[-0,09839 ; -0,00346]`, p-value `0,03898`. Cette qualification reste historique.

L'optimiseur exhaustif a compare les 1 906 884 combinaisons et 5 377 respectaient la perte maximale
de `0,005` hit attendu. La grille sans mise figee pour le 3 aout est `5 6 42 43 44`, Chance `3`.
Son score marginal attendu vaut `0,52073`, contre `0,52542` pour la grille Top-5 pure
`11 20 40 42 44`; la perte est `0,00469`. Son multiplicateur de popularite predit vaut `0,44992`,
contre `1,03512` en moyenne. Cela vise seulement un partage potentiellement moindre en cas de gros
gain et n'augmente pas la probabilite de tirage. La cohorte 4,
`eadab0c5f709aaa8d280cd82d74ae9f628015e2974d6c8ec75f3527cc2033a57`, dispose d'un budget
alpha de `0,0025`. Tete append-only :
`f440b7e043fadcf0e1379299bcd424d93a0617098c5b1d9cf1ae9578e5ee3200`. Empreinte du bundle :
`feb976688a8c238b4f3546ab2b7ad51d98bba1ca608b71e0c6dd6eb813bf0659`.

La version 0.20 remplace l'optimisation ponctuelle de popularite par une borne conservatrice. Cent
modeles de Poisson sont reajustes sur des echantillons en blocs temporels; chaque combinaison est
classee par le quantile 90 % de son multiplicateur, normalise par la popularite moyenne de chaque
echantillon. Les parametres `100`, `0,90`, la taille de bloc et la graine font partie de l'identite
scientifique et le bootstrap est exactement reproductible.

La grille sans mise figee pour le 3 aout est `3 42 43 44 45`, Chance `3`. Son score marginal
attendu vaut `0,52086`, soit une perte de `0,00456` face au Top-5 pur. Son multiplicateur ponctuel
vaut `0,46537` et sa borne conservatrice `0,62204`, contre `0,90605` pour la grille Top-5. La grille
v0.19 aurait une borne `0,63409` avec le meme bootstrap; v0.20 reduit donc le risque d'estimation
qui pilote la selection, sans pretendre augmenter la probabilite de sortie. La cohorte 5,
`67af5224bf5dd2a33799bec286e77a9bbe6728baca3284d2237b71ae44a0e212`, dispose d'un budget
alpha de `0,0016667`. Tete append-only :
`dac620853e531f4f69f595192feab7901ca08829a6c8c0e6e791b6c6e035fe5a`. Empreinte du bundle :
`2a2d41a1468b3043ff0cf4d80de70911f4a3f4c19069b254118b7c272848da7d`.

La version 0.21 ouvre un registre separe pour transformer la qualification historique du modele de
popularite en test prospectif. Les coefficients bruts, le backtest, l'identite scientifique et le
snapshot des donnees sont figes avant le tirage; apres cloture, le score compare la deviance de
Poisson du modele a celle d'une popularite uniforme. Le volume de tickets reste estime par le rang
9. Les hashes sont aussi verifies contre un recalcul integral des metriques, pas uniquement contre
la chaine append-only.

La premiere cohorte exige 100 scores futurs et dispose d'un budget alpha de `0,025`. Son premier
snapshot cible le 3 aout 2026 et reste `insufficient_data` avec zero score. Cohorte :
`4a85eb1f2b0b1a45f253e1c2635498d705d955d84cddfaa91261bca3f499a90e`. Tete append-only :
`82faaae055cd23141698150c15c2bb3b8facfde54e9d65a0bb44f072e6f6f9e0`. Empreinte du bundle :
`bffe038a68f0f19564e8870c0fd0fea06616c4a2e7aece5b3bf555bea7197372`. Cette preuve n'ajoute
aucune grille et ne modifie pas la selection v0.20.

La version 0.22 remplace la cible de selection de foule par `main_combination`. Elle additionne les
gagnants des rangs 1 et 2, soit tous les tickets ayant trouve les cinq numeros principaux, puis
divise le volume estime par `C(49,5)`. Sur les memes 972 observations externes, la deviance vaut
`1,69002` contre `2,25085` pour l'uniforme, delta `-0,56084`, IC temporel
`[-0,84946 ; -0,33202]`, p-value brute `0,000500` et p-value corrigee sur les deux cibles
`0,001000`. Le modele jackpot, le modele combine et les ensembles explores ont ete rejetes comme
remplacements lorsqu'ils ne passaient pas leurs controles corriges.

La grille sans mise figee pour le 3 aout est `33 42 43 44 45`, Chance `3`. Son score marginal
attendu vaut `0,52100`, soit une perte de `0,00441` face au Top-5 pur. Sa popularite principale
ponctuelle vaut `0,24303` et sa borne conservatrice `0,29434`, contre `0,81592` pour le Top-5.
Evaluee avec le meme modele, la grille v0.20 aurait une borne `0,30876`; v0.22 la reduit de
`4,67 %` tout en sacrifiant moins de score marginal. Le numero Chance reste `3` car cette cible
n'apporte aucune validation suffisante pour le remplacer.

La version 0.23 ajoute un diagnostic beta-binomial du partage du numero Chance, sans ouvrir de
nouvelle grille ni depenser le budget d'une cohorte de numeros. Sur 1 749 tirages externes, le
delta de deviance vaut `-0,00888`, mais l'IC temporel `[-0,02199 ; 0,00632]` et la p-value `0,099`
laissent le statut `qualified=false`. Les facteurs bruts des numeros 5 et 7 sont plus eleves,
mais leur instabilite ne justifie pas encore de les eviter.

La version 0.24 ajoute un challenger interactionnel pour la popularite des cinq numeros. Il reduit
la deviance historique a `1,66104` contre `1,69002`, avec delta `-0,58981`, IC
`[-0,88509 ; -0,35812]` et p-value `0,000500`. La borne conservatrice de sa selection actuelle
est toutefois `0,45873`, superieure a `0,29434` pour la v0.22 de base; il reste donc optionnel et
n'ouvre aucune nouvelle grille prospective.
Cohorte challenger : `c87b31a80bde6e4fba5a92ee49f437bb5de7b8044c66ef0494bacea2ad144393`.
Tete du challenger : `b7637fc5f579818fe26508fb190af9c69664a6ed6b23b2d8b04d034c4fd914c7`.
Empreinte du bundle apres preenregistrement :
`fe481c0d43e8f987389aa211c23a14a6cb83ed6628a7fff9a24b940469a63e92`.
Cohorte de foule : `d16157d6de3d4cf94beb79fc07049eab0d64907eb964e3721a4ec4e7d883e3d4`.
Tete de foule : `358b4a959fd8f694f030fdcfb79f73e8259025a20841ca5a39ba99c73de7fe73`.
Cohorte de numeros : `1378a3a51d6a701bcb213c2bc440b9cc01c184f02f02b939b274ea3639bdd4cb`.
Tete des numeros : `72e7d8efab3d809f418eae04e8dca2759e114094bbbd0ddc0bd418bad463b42f`.
Empreinte du bundle de foule :
`db9ac90018768b2bfab2790ca1490d2a6b7473fa07effe83cb6b8145f866d4b9`. Empreinte du bundle
des numeros : `470d9c12f2187c8c76f830e4e77fb7fe07def11e3f5d8f782630951ab754bfeb`.

La version 0.25 ajoute un modele hybride de foule : les six variables structurelles sont completees
par 49 effets regularises indiquant quels numeros composent la grille. Contrairement au modele de
tirage, ces effets cherchent uniquement les preferences des joueurs. Sur 972 observations externes,
la deviance tombe a `1,64870`, delta `-0,60215` contre l'uniforme, IC par blocs
`[-0,94564 ; -0,34970]`. La p-value brute `0,000500` devient `0,00300` apres correction des deux
cibles et des trois schemas explores.

Avec les probabilites marginales deja figees pour le 3 aout, le diagnostic exhaustif selectionne
`1 42 43 44 45`, Chance `3`, pour une perte de hit attendu de `0,00455`. Sa popularite ponctuelle
vaut `0,22058` et sa borne a 90 % `0,28211`, contre `0,29434` pour la grille v0.22. Cette grille
n'est pas substituee a la prevision publique du 3 aout : le nouveau schema devient le defaut des
prochaines previsions seulement.

Le premier essai prospectif v0.25 cible le 5 aout : `1 42 43 44 45`, Chance `9`. Sa borne de
popularite a 90 % vaut `0,27941`, contre `0,98480` pour le Top-5 pur, avec une perte de hit attendu
de `0,00455`. Le classement des numeros reste `forced_experimental` et ne revendique aucun signal
sur les boules. La cohorte de foule `2eb9c12d1475ae60a56439d94210b56d9ba4bbafc7257f4ca6a1c13aa7a1f9a5`
et la cohorte de numeros `fd1c3a29e89c40fed8555d3d4280fe8c1c7a04b280853c42aefa795eedaf4542`
restent toutes deux `insufficient_data`. Tetes append-only :
`2182a53a50cbf1e9a762d6a801174282633df5259b90d18e0be504b47689cd6e` pour la foule et
`3ca9634dd7055c0a9507cce5cb46a882aa49d63bca4cbad4d3466c3cfd0d2c42` pour les numeros.
Empreintes des bundles :
`b3f3c9394761bff41fea920be123603c6abef6c3572796dd3d8718d5b6f98ffc` et
`221e520a8bd4d8eed200a837559120627a9db24583564b246c6d5a1dc818d285`.

A 17 h 24, heure de Paris, le meme challenger a aussi ete fige avant le tirage du 3 aout, sans
remplacer les previsions precedentes : `1 42 43 44 45`, Chance `3`. Sa borne de popularite vaut
`0,28211`, contre `0,98131` pour le Top-5 pur. Tetes append-only :
`0d5a9732bb72268734342368faf55edb9c553ccd53192d01c2b0db9d234c392a` pour la foule et
`e1fad88b62198d87718ba81cebf439ad641e2abff8747a483b2a6758d4269032` pour les numeros.
Empreintes des bundles avant tirage :
`c5c6e86349083ef27d283c68b5015f6ae441837b3d1495c12cdc5329ba72f575` et
`03cb68727a5ce0eae54ff33fc919ef1e465525bd1853e65597343ea69108f95f`.
