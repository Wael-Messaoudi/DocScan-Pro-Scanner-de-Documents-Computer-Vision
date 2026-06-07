# DocScan Pro
[![Ask DeepWiki](https://devin.ai/assets/askdeepwiki.png)](https://deepwiki.com/Wael-Messaoudi/DocScan-Pro-Scanner-de-Documents-Computer-Vision)

**DocScan Pro** est une application de bureau pour Windows, macOS et Linux qui transforme des photos de documents en images numérisées de haute qualité. Utilisant des techniques classiques de vision par ordinateur avec OpenCV, l'application offre un pipeline complet pour redresser la perspective, améliorer le contraste et nettoyer les images, ainsi que des outils pour explorer individuellement divers filtres et algorithmes de traitement d'image.

L'interface, développée avec Tkinter, propose une expérience utilisateur simple et intuitive avec un thème sombre.

## Fonctionnalités

### Scanner de documents complet
- **Détection automatique du document :** Lance un pipeline complet qui identifie les bords du document dans une image.
- **Correction de perspective :** Redresse l'image pour obtenir une vue de dessus parfaite.
- **Amélioration de l'image :** Applique un seuillage adaptatif pour créer un effet de scanner noir et blanc à fort contraste, améliorant la lisibilité.
- **Visualisation étape par étape :** L'onglet "Scanner" affiche l'image originale, le contour détecté et le résultat final côte à côte.

### Traitements d'image individuels
Explorez et appliquez une large gamme de techniques de vision par ordinateur sur n'importe quelle image chargée :
- **Filtres :** Flou Gaussien, Flou Médian, Filtre Bilatéral pour la réduction du bruit.
- **Détection de contours :** Algorithmes de Canny et Sobel pour extraire les bords.
- **Seuillage :** Méthodes globale, adaptative et Otsu pour binariser les images.
- **Opérations Morphologiques :** Érosion, Dilatation, Ouverture et Fermeture pour manipuler les formes des objets.
- **Détection de Caractéristiques :** Identification de points d'intérêt avec les coins de Harris et la détection de blobs.
- **Masquage ROI :** Isolez une région d'intérêt rectangulaire au centre de l'image.

### Interface utilisateur
- **Chargement et sauvegarde :** Chargez des images depuis votre disque local (`.png`, `.jpg`, `.bmp`) et sauvegardez le résultat.
- **Exportation PDF :** Exportez directement l'image numérisée ou traitée en tant que fichier PDF.
- **Journal d'analyse :** Un onglet "Analyse" enregistre toutes les opérations effectuées, les dimensions des images et les temps de traitement pour chaque étape.
- **Feedback en temps réel :** Une barre de statut et une barre de progression informent l'utilisateur de l'opération en cours.

## Le pipeline du Scanner
Le processus de numérisation automatique en un clic suit un pipeline optimisé :

1.  **Chargement & Redimensionnement :** L'image est chargée et redimensionnée pour un traitement plus rapide tout en conservant son ratio.
2.  **Pré-traitement :** Conversion en niveaux de gris et application d'un filtre Gaussien pour réduire le bruit.
3.  **Détection de Contours (Canny) :** L'algorithme de Canny est utilisé pour trouver les bords nets de l'image.
4.  **Opérations Morphologiques :** Une dilatation suivie d'une érosion aide à fermer les possibles discontinuités dans les contours du document.
5.  **Recherche du Contour :** Le plus grand contour quadrilatère est identifié comme étant le document. Un mécanisme de repli est prévu si aucun contour n'est trouvé.
6.  **Correction de Perspective :** L'image est "déformée" à l'aide d'une transformation de perspective à quatre points pour obtenir une vue de dessus rectangulaire.
7.  **Seuillage Adaptatif Final :** Un seuillage adaptatif est appliqué sur l'image redressée pour produire l'effet final "scanné", net et contrasté.

## Technologies utilisées
- **Python 3**
- **OpenCV** pour toutes les opérations de vision par ordinateur.
- **Tkinter** pour l'interface graphique.
- **Pillow (PIL)** pour la manipulation d'images et leur intégration dans Tkinter.
- **NumPy** pour les calculs numériques et la manipulation de matrices.

## Installation et lancement

### Prérequis
- Python 3.6 ou supérieur.
- pip (généralement inclus avec Python).

### Installation

1.  Clonez le dépôt sur votre machine locale :
    ```bash
    git clone https://github.com/Wael-Messaoudi/DocScan-Pro-Scanner-de-Documents-Computer-Vision.git
    cd DocScan-Pro-Scanner-de-Documents-Computer-Vision
    ```

2.  Installez les dépendances Python requises :
    ```bash
    pip install opencv-python numpy Pillow
    ```

### Lancement

Exécutez `app.py` pour démarrer l'application :
```bash
python app.py
```
L'application se lancera, vous permettant de charger une image et de commencer à utiliser ses fonctionnalités.

## Structure du projet

-   `app.py`: Point d'entrée de l'application. Il initialise et lance l'interface graphique.
-   `gui.py`: Contient l'intégralité du code de l'interface utilisateur (fenêtres, boutons, panneaux d'images, onglets). Il gère les événements et appelle les fonctions de traitement correspondantes depuis `scanner.py`.
-   `scanner.py`: Cœur de l'application. Ce module contient toutes les fonctions de traitement d'image basées sur OpenCV, du simple filtre au pipeline de numérisation complet. Chaque fonction est optimisée pour fonctionner indépendamment.
