# Mesures de référence : dgemm_seq avant blocage et vectorisation

Objectif : mesurer l'effet des options du compilateur sans modifier le code.

## Conditions
- Machine : miriel042, Xeon E5-2680 v3 (Haswell, 2,5 GHz), pic = 40 GFlop/s par coeur
- 1 seul thread (OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1), médiane de 5 mesures
- Code mesuré : dgemm_seq.c (ordre n,k,m, sans blocage ni vectorisation manuelle),
  copie dans dgemm_seq_mesure.c
- Builds : o2 = -O2 ; o3 = -O3 ; o3-haswell = -O3 -march=haswell

## Résultats (GFlop/s, n = 1000)
| Build | GFlop/s | % du pic |
|---|---|---|
| -O2 | 2,06 | 5 % |
| -O3 | 3,66 | 9 % |
| -O3 -march=haswell | 4,82 | 12 % |
| OpenBLAS (référence) | 34,8 | 87 % |

## Correction
Tous les tests passent sur les 3 builds (500 tests check_dgemm, 25 tests check_dgetrf,
17 tailles/transpositions difficiles) : voir correction.txt.

## Points à noter
- n = 512 et 1024 sont 15 à 18 % plus lents que les tailles voisines (conflits de cache).
- La performance baisse à n = 1500 (matrices > cache L3) ; à confirmer en répétant la mesure
  sur un noeud peu chargé.

## Contenu du dossier
- environnement.txt : machine, processeur, caches, gcc, charge
- dgemm_seq_mesure.c : code mesuré
- correction.txt : verdicts de correction
- dgemm_flags.csv : tableau principal (builds x tailles, plus OpenBLAS)
- dgemm_puissance2.csv : effet des puissances de 2
- resume.txt : GFlop/s et % du pic
Relancer : bash tests_flags.sh <nom_du_dossier> depuis la racine du projet.
