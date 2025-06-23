# Knowledge Distillation for Document Classification

Ce projet implémente la distillation de connaissances pour la classification de documents en utilisant le dataset RVL-CDIP avec des fonctionnalités enrichies (OCR, bounding boxes).

## Architecture

- **Teacher Model**: DiT (Document Image Transformer) - `microsoft/dit-large-finetuned-rvlcdip`
- **Student Model**: LayoutLMv3-Tiny (version compacte personnalisée)
- **Dataset**: RVL-CDIP avec features OCR et bounding boxes

## Structure du Projet

```
├── config.py          # Configuration et hyperparamètres
├── models.py          # Définition des modèles teacher et student
├── dataset.py         # Classe dataset avec features enrichies
├── losses.py          # Fonctions de loss pour la distillation
├── train.py           # Boucle d'entraînement et évaluation
├── utils.py           # Fonctions utilitaires
├── main.py            # Script principal
├── requirements.txt   # Dépendances Python
└── README.md         # Documentation
```

## Fonctionnalités

### Dataset Enhanced
Le dataset utilise les features suivantes :
- `image`: Image PIL du document
- `width`, `height`: Dimensions originales
- `category`: Nom de catégorie 
- `ocr_words`: Mots extraits par OCR
- `word_boxes`: Bounding boxes des mots `[xmin, ymin, xmax, ymax]`
- `ocr_paragraphs`: Paragraphes extraits
- `paragraph_boxes`: Bounding boxes des paragraphes
- `label`: Label numérique (0-15)

### Models

#### Teacher (DiT)
- Modèle complexe pré-entraîné sur RVL-CDIP
- Traitement image seule
- Gelé pendant l'entraînement (pas de mise à jour des poids)

#### Student (LayoutLMv3-Tiny)
- Version compacte de LayoutLMv3
- Paramètres réduits :
  - Hidden size: 768 → 256
  - Layers: 12 → 4
  - Attention heads: 12 → 4
  - Intermediate size: 3072 → 1024
- Utilise texte OCR + layout + image

### Distillation
- **Alpha**: 0.7 (70% distillation, 30% supervision directe)
- **Temperature**: 4.0 (adoucissement des probabilités)
- **Loss**: Combinaison KL Divergence + Cross Entropy

## Installation

```bash
# Cloner le repo
git clone <repo_url>
cd knowledge-distillation

# Installer les dépendances
pip install -r requirements.txt
```

## Utilisation

### Entraînement

```bash
# Entraînement standard
python main.py --mode train

# Avec paramètres personnalisés
python main.py --mode train --lr 2e-4 --epochs 10 --alpha 0.8 --temperature 5.0
```

### Évaluation

```bash
# Évaluer le meilleur modèle
python main.py --mode eval

# Évaluer un modèle spécifique
python main.py --mode eval --model path/to/model.pth
```

### Informations Dataset

```bash
# Afficher les informations du dataset
python main.py --mode info
```

### Test Rapide

Pour tester le fonctionnement, décommentez la ligne dans `main.py` :
```python
# test_single_batch()  # Décommenter pour tester
```

## Configuration

Les hyperparamètres peuvent être modifiés dans `config.py` :

```python
BATCH_SIZE = 16
NUM_EPOCHS = 6
LEARNING_RATE = 3e-4
ALPHA = 0.7
TEMPERATURE = 4.0
NUM_CLASSES = 16
```

## Sauvegarde/Reprise

Le système sauvegarde automatiquement :
- **Checkpoint complet** : `latest_checkpoint.pth` (pour reprendre l'entraînement)
- **Meilleur modèle** : `student_model.pth` (pour l'inférence)

Pour reprendre l'entraînement :
```bash
python main.py --mode train --checkpoint latest_checkpoint.pth
```

## Fonctionnalités Avancées

### Monitoring
- Barres de progression avec métriques temps réel
- Tracking des losses (Total, Cross-Entropy, Knowledge Distillation)
- Précision train/validation

### Debugging
- Messages d'erreur détaillés
- Debug des échantillons dataset
- Vérification compatibilité modèles

### Utilitaires
- Gestion seeds reproductibilité
- Informations GPU/mémoire
- Comptage paramètres modèles
- Logging des expériences

## Exemple de Sortie

```
=== Démarrage du processus de distillation de connaissances ===
Configuration:
  Mode: train
  Batch size: 16
  Epochs: 6
  Learning rate: 0.0003
  Alpha (distillation weight): 0.7
  Temperature: 4.0

=== Chargement des modèles ===
Teacher model loaded: microsoft/dit-large-finetuned-rvlcdip

Teacher (DiT) Model Info:
  Total parameters: 303,535,056
  Trainable parameters: 0
  Model size: ~1,158.6 MB

Student model created: LayoutLMv3-Tiny
Parameters: ~25,234,832

Student (LayoutLMv3-Tiny) Model Info:
  Total parameters: 25,234,832
  Trainable parameters: 25,234,832
  Model size: ~96.3 MB

=== Chargement du dataset ===
Chargement du dataset jinhybr/rvl_cdip_400_train_val_test...
Taille du dataset: 6400
Colonnes disponibles: ['image', 'width', 'height', 'category', 'ocr_words', 'word_boxes', 'ocr_paragraphs', 'paragraph_boxes', 'label']
```

## Optimisations Futures

1. **Attention Distillation** : Transférer les cartes d'attention
2. **Feature Matching** : Loss sur les features intermédiaires
3. **Progressive Growing** : Augmenter graduellement la taille du student
4. **Multi-Teacher** : Combiner plusieurs teachers
5. **Quantization** : Post-training quantization du student

## Troubleshooting

### Erreurs Communes

1. **Mémoire GPU insuffisante** : Réduire `BATCH_SIZE` dans `config.py`
2. **Erreur OCR processing** : Le code a des fallbacks pour traiter image seule
3. **Checkpoint incompatible** : Supprimer `latest_checkpoint.pth` pour redémarrer

### Debug

Pour activer le debug détaillé, modifier les print statements dans `dataset.py` :
```python
# Décommenter les lignes DEBUG dans __getitem__
print(f"DEBUG: Début __getitem__ pour l'index {idx}")
```

## Performances Attendues

Avec la configuration par défaut :
- **Teacher (DiT)** : ~95% précision
- **Student sans distillation** : ~75% précision  
- **Student avec distillation** : ~85% précision (gain de ~10%)

La distillation permet d'obtenir un modèle **~12x plus petit** avec seulement **~10% de perte de précision**. 
