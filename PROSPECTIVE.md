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
