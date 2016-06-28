# DataforGood

Frateli met en contact des lycées de milieux modestes avec des parrains pour les guider tout au long de leurs études. Notre but est de voir ce qui fait qu'un parrainage va marcher ou pas.

*En raison du caractère confidentiel des données, celles-ci ne sont pas présente dans ce repositery.*

## Travail effectué

Nous avons opté pour le choix de la métrique [AUC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) en raison du jeux de données déséquilibré ainsi que pour fournir une probabilité de réussite d'un parrainage à l'équipe de Frateli.

On peux interprèter cette valeur comme la probabilité de classer un exemple positif (un pairranage réussi) choisi au harsard comme réellement positif. Cette mesure va de 0 à 1 (1 étant un classificateur parfait).

Résultat du modèle sur une cross-validation à 5 fold est de 0.729 AUC +/- 0.046 (screen-shot)

![Image Evaluation model](https://cloud.githubusercontent.com/assets/8374843/16419159/00947074-3d4d-11e6-8066-a6e1ae73ff74.png)

On est arrivé à plus de 0.75 AUC en combinant plusieurs modèles et features, mais celà mène à une complexité non négligeable par la suite.
