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
