# GeneticScheduler

L'idée de l'algorithme est de donner un ordre d'execution de taches sur plusieurs CPUs sachant que les tâches dependent les unes des autres.

Pour résoudre ce problème on utilise un algorithme genetique codé en parallele avec MPI

## Données

L'algorithme utilise un fichier JSON comme des données, et un bon fichier JSON doit être sous le format suivant :
```
{
  "nodes": {
    "47": {
      "Data": "00:27:21.7575934",
      "Dependencies": []
    },
    "48": {
      "Data": "00:45:53.3954572",
      "Dependencies": [
          47
      ]
    },
    "49": {
      "Data": "00:35:41.0118642",
      "Dependencies": [
        48
      ]
    }
  }
```

## Lancer l'algorithme

Pour lancer l'algorithme, il faut changer les paramètres dans les scripts, dont ```code/script.py``` est le script pour la méthode de calcul sur un seul coeur, et ```code/mpi_main.py``` est pour la méthode parallélisée. Après, n'oubliez pas uncommenter le code sous ```if __name__=="__main__"```. Lancez dans un terminal comme un exemple :

```
$ python code/script.py                     # Pour le script.py
$ mpiexec -np 4 python code/mpi_main.py     # Pour la méthode parallélisée
```

## L'algorithme

Pour plus de details de cet algorithme, veuillez lire le rapport suivant en cliquant [rapport](https://teams.microsoft.com/_#/docx/viewer/aggregatefiles/https:~2F~2Faneoconsulting.sharepoint.com~2Fsites~2FEXTCENTRALESUPELEC~2FShared%2520Documents~2FGeneral~2FGroupe%25201~2FRapport%2520Final.docx?baseUrl=https:~2F~2Faneoconsulting.sharepoint.com~2Fsites~2FEXTCENTRALESUPELEC&fileId=3ce82c65-2030-4877-86ba-c159edc2d9d4&ctx=aggregate&viewerAction=view)

