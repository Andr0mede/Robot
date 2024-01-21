#!/bin/bash

# Récupérez le répertoire du script
script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Vérifiez le nombre d'arguments
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <nombre_de_joueurs> <ID_de_map>"
    exit 1
fi

# Récupérez le nombre de joueurs et l'ID de map depuis les arguments
nb_joueurs=$1
id_map=$2

# Répertoire du jeu
jeu_dossier="$script_dir"

# Lancez le serveur dans le premier terminal
gnome-terminal --tab --title="Serveur" -- bash -c "cd $jeu_dossier; python3 server.py -nb $nb_joueurs -mi $id_map; exec bash"

# Attendez un peu pour laisser le serveur démarrer
sleep 4

# Lancez les agents dans les terminaux suivants
for ((i=1; i<=$nb_joueurs; i++)); do
    gnome-terminal --tab --title="Agent $i" -- bash -c "cd $jeu_dossier; python3 agent.py; exec bash"
    sleep 1
done

exit 0

