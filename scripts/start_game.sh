#!/bin/bash

# Vérifiez le nombre d'arguments
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <nombre_de_joueurs>"
    exit 1
fi

# Récupérez le nombre de joueurs depuis le premier argument
nb_joueurs=$1

# Répertoire du jeu
jeu_dossier="/home/andro/SynologyDrive/Drive/Ecole/CDI/In512\ Systèmes\ intelligents\ distribués/Projet/scripts"

# Lancez le serveur dans le premier terminal
gnome-terminal --tab --title="Serveur" --command="bash -c 'cd $jeu_dossier; python3 server.py -nb $nb_joueurs; exec bash'"

# Attendez un peu pour laisser le serveur démarrer
sleep 2

# Lancez les agents dans les terminaux suivants
for ((i=1; i<=$nb_joueurs; i++)); do
    gnome-terminal --tab --title="Agent $i" --command="bash -c 'cd $jeu_dossier; python3 agent.py; exec bash'"
done

exit 0

