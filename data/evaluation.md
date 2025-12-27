# Copilot Chat Conversation Export: Evaluation framework for skin cancer detection

**User:** @marksamfd
**Thread URL:** https://github.com/copilot/c/ee4e9149-4c0f-4987-9d14-94baf2811a07

## @marksamfd

Develop Evaluation Framework: Create a "ground truth" evaluation set of questions categorized by complexity: Category 1: Simple Lookup (e.g., "Find the User class.") Category 2: Local Context (e.g., "What variables does this function use?") Category 3: Global/Relational (e.g., "Trace the full user authentication flow.") based on these give me about 10 questions for each category and their answers answer both in JSON and MD format

## Copilot

I'll analyze your repository to understand its structure and create an evaluation framework with questions appropriate for a skin cancer detection project using ML/DL with XAI techniques. Now I have a comprehensive understanding of the repository. Let me create the evaluation framework with questions categorized by complexity.

# Evaluation Framework:  Ground Truth Question Set

Based on my analysis of the **Explainable-AI-for-Skin-Cancer-Detection** repository, here is a comprehensive evaluation set of questions categorized by complexity. 

---

## Markdown Format

### Category 1: Simple Lookup (Direct Code/Definition Queries)

| # | Question | Answer |
|---|----------|--------|
| 1 | Find the `SkinLesionCNN` class. | Located in `XAI/modeling/models/SkinLesionCNN.py`. It is a CNN model for skin lesion classification based on the paper "Skin lesion classification of dermoscopic images using machine learning and convolutional neural network".  |
| 2 | Where is the `BaseModel` class defined? | Located in `XAI/modeling/models/Base_Model.py`. It extends `torch.nn.Module` and provides an abstract base class for all deep learning models. |
| 3 | Find the `CLASS_NAMES` dictionary. | Located in `XAI/config. py` (lines 37-45). Contains 7 skin lesion classes:  akiec, bcc, bkl, df, mel, nv, vasc mapped to their full names. |
| 4 | Where is the `HAM10000Dataset` class located? | Located in `XAI/dataset. py` (lines 106-150). It's a PyTorch Dataset class for loading HAM10000 skin lesion images. |
| 5 | Find the `plot_confusion_matrix` function. | Located in `XAI/plots. py` (lines 177-223). Creates visualizations of confusion matrices with optional normalization. |
| 6 | Where is the `BATCH_SIZE` configuration defined? | Located in `XAI/config.py` (line 31). Default value is 32. |
| 7 | Find the `HairRemoval` preprocessing class. | Located in `XAI/preprocessing/hair_removal.py`. A preprocessing transform for removing hair artifacts from dermoscopic images. |
| 8 | Where is the `dl_models` list defined? | Located in `XAI/modeling/AllModels.py` (lines 20-24). Contains all available deep learning model classes. |
| 9 | Find the `DTModel` (Decision Tree Model) class. | Located in `XAI/modeling/models/DecisionTreeModel.py`. Implements a Decision Tree classifier for skin lesion classification. |
| 10 | Where is the `MODELS_DIR` path configured? | Located in `XAI/config.py` (line 15). Points to `ROOT_DIR / "models"` for storing trained model checkpoints. |

---

### Category 2: Local Context (Variable/Function Usage Queries)

| # | Question | Answer |
|---|----------|--------|
| 1 | What variables does the `train_model` function use for tracking training progress? | Uses `best_val_loss` (initialized to infinity), `history` dictionary containing lists for `train_loss`, `train_acc`, `val_loss`, `val_acc`, and `lr`, and a `SummaryWriter` for TensorBoard logging (XAI/modeling/train. py, lines 86-88). |
| 2 | What transforms are applied in the `get_transforms("train")` function? | Applies HairRemoval, CLAHE, EnhanceClarityCV, v2. ToImage, RandomHorizontalFlip(p=0.5), RandomVerticalFlip(p=0.5), RandomRotation(25), ToDtype(torch.float32), and Normalize with ImageNet means/stds (XAI/dataset.py, lines 163-178). |
| 3 | What parameters does the `predict_image` function return? | Returns a tuple of `(predicted_class, class_name, probabilities)` - the class index, human-readable class name from CLASS_NAMES, and numpy array of softmax probabilities (XAI/modeling/predict.py, lines 100-138). |
| 4 | What layers are defined in the `SkinCancerCNN.__init__` method? | Defines `block1` (Conv2d→ReLU→Conv2d→ReLU→MaxPool2d→Dropout), `block2` (similar with 64 channels), and subsequent blocks with increasing channel depth, plus a classifier (XAI/modeling/models/custom_cnn_2. py, lines 13-28). |
| 5 | What configuration variables does the `prepare_data` function use? | Uses `HAM10000_METADATA`, `CLASS_NAMES`, `RANDOM_SEED`, `BATCH_SIZE`, and `HAM10000_IMAGES_PART1` from XAI/config. py (XAI/dataset. py, lines 195-296). |
| 6 | What metrics does `test_model` calculate and return? | Calculates and returns `accuracy`, `classification_report` (dict), `confusion_matrix`, `predictions` array, and `true_labels` array (XAI/modeling/train.py, lines 317-323). |
| 7 | What arguments does the `explain_prediction_lime` function accept? | Accepts `model`, `image`, `transform=None`, `num_samples=1000`, and `save_path=None` (XAI/modeling/predict.py, lines 236-249). |
| 8 | What feature extraction functions are called in `extract_all_features`? | Calls `extract_color_histogram`, `extract_shape_features`, `extract_texture_features`, `extract_lbp_features`, and `extract_glcm_features` (XAI/features.py). |
| 9 | What hyperparameters does `plot_tsne_features` use for t-SNE?  | Uses `perplexity=30`, `n_iter=1000`, `n_components=2`, and `random_state=42` (XAI/plots.py, lines 323-360). |
| 10 | What loss functions are supported in the `main` training function? | Supports `nn.BCEWithLogitsLoss()` for binary classification and `nn.CrossEntropyLoss()` for multi-class classification, selected based on `isBinary` flag (XAI/modeling/train.py, line 421). |

---

### Category 3: Global/Relational (Cross-File Flow Queries)

| # | Question | Answer |
|---|----------|--------|
| 1 | Trace the full data loading and preprocessing pipeline from download to DataLoader.  | 1) `download_and_extract_ham10000()` downloads dataset via Kaggle API → 2) `organize_data()` sorts images by class into `INTERIM_DATA_DIR` → 3) `prepare_data()` loads metadata, optionally balances classes, splits into train/val/test → 4) Creates `HAM10000Dataset` instances with transforms from `get_transforms()` → 5) Returns PyTorch DataLoaders with configured batch size and workers.  Flow:  XAI/dataset. py lines 37-296. |
| 2 | Trace the complete model training flow from initialization to saving. | 1) `main()` calls `set_seed()` → 2) `prepare_data()` loads data → 3) Model instantiated via `ResizedModel(dl_models[i]. inputSize(), dl_models[i]())` → 4) `load_best_model()` checks for existing checkpoint → 5) `train_model()` runs training loop with optimizer/scheduler → 6) `evaluate_model()` computes validation metrics each epoch → 7) Best model saved to `MODELS_DIR` with val_acc in filename → 8) `test_model()` evaluates final performance.  Flow spans XAI/modeling/train.py and XAI/modeling/AllModels.py.  |
| 3 | How do the XAI explanation methods (LIME, SHAP, GradCAM) integrate with predictions? | 1) Model loaded via `load_best_model()` → 2) `predict_image()` generates predictions → 3) For LIME: `explain_prediction_lime()` uses `lime_image. LimeImageExplainer` with custom `predict_fn` wrapper → 4) For SHAP: `explain_prediction_shap()` uses `shap. DeepExplainer` with background tensor → 5) For GradCAM:  `explain_prediction_gcam()` uses `pytorch_grad_cam.GradCAM` with target layers from `get_target_layers()`. All methods in XAI/modeling/predict.py, with ML-specific explainers in XAI/explainability/.  |
| 4 | Trace how configuration values flow from config. py through the entire codebase. | `XAI/config.py` defines paths (ROOT_DIR, DATA_DIR, MODELS_DIR), model params (MODEL_INPUT_SIZE, BATCH_SIZE, NUM_EPOCHS), and class mappings (CLASS_NAMES, NUM_CLASSES). These are imported by:  `dataset.py` (data paths, batch size), `train.py` (training params, model paths), `predict.py` (class names, paths), `plots.py` (class names, figure paths), `features.py` (data paths), and all model files (NUM_CLASSES). |
| 5 | How does the preprocessing pipeline connect to model input requirements? | 1) `XAI/preprocessing/` contains HairRemoval, CLAHE, EnhanceClarityCV, ContrastStretch → 2) `get_transforms()` in dataset.py chains these with torchvision.transforms.v2 → 3) `HAM10000Dataset.__getitem__` applies transform pipeline → 4) `ResizeLayer. py` contains `ResizedModel` wrapper that resizes input to model's `inputSize()` → 5) Each model class defines `inputSize()` static method (e.g., 224x224). Flow: preprocessing → dataset. py → modeling/ResizeLayer.py → model.  |
| 6 | Trace how feature extraction connects ML models to explainability. | 1) `XAI/features.py` defines `extract_all_features()` combining color, shape, texture, LBP, GLCM features → 2) Features saved to PROCESSED_DATA_DIR → 3) `train_ml. py` loads features, scales via StandardScaler, trains DTModel/RFModel → 4) `XAI/explainability/shap_ml_explainer.py` uses same features for SHAP explanations → 5) `XAI/explainability/lime_ml_explainer. py` uses features for LIME → 6) `run_explainers. py` orchestrates all ML explanations with `analyze_image()`. |
| 7 | How do the different model architectures (CNN variants) share common base functionality? | 1) `Base_Model.py` defines abstract `BaseModel(nn.Module)` with `name()` and `inputSize()` abstract methods → 2) All CNN models inherit:  SkinLesionCNN, CustomCNN, SkinCancerCNN, D2CNN, ModifiedInceptionV3, FineTunedResNet50, SkinEfficientNetB5, MobileNetV2 → 3) `AllModels.py` collects all model classes in `dl_models` list → 4) `train. py` and `predict.py` iterate through this list for training/evaluation → 5) ML models use separate `ML_Base_model.py` hierarchy.  |
| 8 | Trace the complete explainability report generation flow in run_explainers. py. | 1) `main()` parses CLI args → 2) `analyze_image()` loads model via `load_latest_model()`, scaler via `load_feature_scaler()` → 3) `load_background_data()` prepares reference data → 4) `explain_image_with_feature_importance()` generates feature rankings → 5) `explain_prediction_with_lime()` + `visualize_lime_explanation()` → 6) `explain_prediction_with_shap()` + `visualize_shap_explanations()` → 7) `explain_top_features_for_image()` for PDP → 8) `generate_combined_report()` creates HTML summary. |
| 9 | How does checkpoint loading and resuming work across training runs? | 1) `load_best_model()` in train.py scans MODELS_DIR for `{model_name}*. pth` files → 2) Parses val_acc from filename pattern `{name}-{val_acc}-e{epoch}-{timestamp}. pth` → 3) Returns best checkpoint path and data → 4) `main()` loads `model_state_dict` and `optimizer_state_dict` from checkpoint → 5) Sets `start_epoch = checkpoint['epoch'] + 1` and `best_val_acc = checkpoint['val_acc']` → 6) `train_model()` continues from `start_epoch` with preserved best accuracy threshold.  |
| 10 | Trace how class balancing affects the entire pipeline from data to evaluation. | 1) `prepare_data(balanced=True)` in dataset.py → 2) Counts class distribution via `metadata['dx']. value_counts()` → 3) Finds `min_class_count` across all 7 classes → 4) Downsamples larger classes with `class_df.sample(min_class_count)` → 5) Balanced DataFrame used for train/val/test split with `stratify=metadata['dx']` → 6) `HAM10000Dataset` maps classes to indices via `class_to_idx` → 7) `test_model()` generates per-class metrics in classification_report → 8) `plot_confusion_matrix()` visualizes class-wise performance.  |

---

## JSON Format

```json
{
  "evaluation_framework": {
    "repository": "marksamfd/Explainable-AI-for-Skin-Cancer-Detection",
    "description": "Ground truth evaluation set for code understanding and retrieval",
    "categories": {
      "category_1_simple_lookup": {
        "description": "Direct code/definition queries - finding specific classes, functions, or variables",
        "questions":  [
          {
            "id": 1,
            "question": "Find the SkinLesionCNN class.",
            "answer":  {
              "location": "XAI/modeling/models/SkinLesionCNN.py",
              "description": "CNN model for skin lesion classification based on the paper 'Skin lesion classification of dermoscopic images using machine learning and convolutional neural network'",
              "lines": "1-11"
            }
          },
          {
            "id":  2,
            "question": "Where is the BaseModel class defined?",
            "answer": {
              "location": "XAI/modeling/models/Base_Model.py",
              "description": "Abstract base class extending torch.nn.Module for all deep learning models",
              "lines": "1-287"
            }
          },
          {
            "id": 3,
            "question": "Find the CLASS_NAMES dictionary.",
            "answer":  {
              "location": "XAI/config.py",
              "description":  "Contains 7 skin lesion classes:  akiec, bcc, bkl, df, mel, nv, vasc mapped to their full names",
              "lines":  "37-45"
            }
          },
          {
            "id": 4,
            "question": "Where is the HAM10000Dataset class located?",
            "answer": {
              "location":  "XAI/dataset.py",
              "description": "PyTorch Dataset class for loading HAM10000 skin lesion images",
              "lines": "106-150"
            }
          },
          {
            "id": 5,
            "question": "Find the plot_confusion_matrix function.",
            "answer": {
              "location": "XAI/plots.py",
              "description": "Creates visualizations of confusion matrices with optional normalization",
              "lines": "177-223"
            }
          },
          {
            "id": 6,
            "question":  "Where is the BATCH_SIZE configuration defined?",
            "answer": {
              "location": "XAI/config. py",
              "description": "Default value is 32",
              "lines":  "31"
            }
          },
          {
            "id": 7,
            "question":  "Find the HairRemoval preprocessing class.",
            "answer":  {
              "location": "XAI/preprocessing/hair_removal.py",
              "description": "Preprocessing transform for removing hair artifacts from dermoscopic images"
            }
          },
          {
            "id": 8,
            "question": "Where is the dl_models list defined?",
            "answer": {
              "location": "XAI/modeling/AllModels. py",
              "description": "Contains all available deep learning model classes",
              "lines": "20-24"
            }
          },
          {
            "id":  9,
            "question": "Find the DTModel (Decision Tree Model) class.",
            "answer":  {
              "location": "XAI/modeling/models/DecisionTreeModel. py",
              "description": "Implements a Decision Tree classifier for skin lesion classification"
            }
          },
          {
            "id": 10,
            "question": "Where is the MODELS_DIR path configured?",
            "answer": {
              "location":  "XAI/config.py",
              "description": "Points to ROOT_DIR / 'models' for storing trained model checkpoints",
              "lines":  "15"
            }
          }
        ]
      },
      "category_2_local_context": {
        "description": "Variable/function usage queries - understanding local scope and parameters",
        "questions": [
          {
            "id":  1,
            "question": "What variables does the train_model function use for tracking training progress? ",
            "answer": {
              "location": "XAI/modeling/train.py",
              "variables": [
                "best_val_loss (initialized to infinity)",
                "history (dict with train_loss, train_acc, val_loss, val_acc, lr lists)",
                "SummaryWriter for TensorBoard logging"
              ],
              "lines": "86-88"
            }
          },
          {
            "id":  2,
            "question": "What transforms are applied in the get_transforms('train') function?",
            "answer":  {
              "location": "XAI/dataset.py",
              "transforms": [
                "HairRemoval()",
                "CLAHE()",
                "EnhanceClarityCV()",
                "v2.ToImage()",
                "RandomHorizontalFlip(p=0.5)",
                "RandomVerticalFlip(p=0.5)",
                "RandomRotation(25)",
                "ToDtype(torch.float32, scale=True)",
                "Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])"
              ],
              "lines": "163-178"
            }
          },
          {
            "id": 3,
            "question": "What parameters does the predict_image function return?",
            "answer": {
              "location": "XAI/modeling/predict.py",
              "returns": {
                "tuple": "(predicted_class, class_name, probabilities)",
                "predicted_class": "int - class index",
                "class_name": "str - human-readable class name from CLASS_NAMES",
                "probabilities": "numpy array of softmax probabilities"
              },
              "lines":  "100-138"
            }
          },
          {
            "id": 4,
            "question": "What layers are defined in the SkinCancerCNN.__init__ method?",
            "answer": {
              "location": "XAI/modeling/models/custom_cnn_2.py",
              "layers": [
                "block1: Conv2d→ReLU→Conv2d→ReLU→MaxPool2d→Dropout (32 channels)",
                "block2: Similar structure with 64 channels",
                "Subsequent blocks with increasing channel depth",
                "classifier:  Fully connected layers for classification"
              ],
              "lines": "13-28"
            }
          },
          {
            "id":  5,
            "question": "What configuration variables does the prepare_data function use? ",
            "answer": {
              "location": "XAI/dataset.py",
              "config_imports": [
                "HAM10000_METADATA",
                "CLASS_NAMES",
                "RANDOM_SEED",
                "BATCH_SIZE",
                "HAM10000_IMAGES_PART1"
              ],
              "lines": "195-296"
            }
          },
          {
            "id": 6,
            "question":  "What metrics does test_model calculate and return?",
            "answer": {
              "location": "XAI/modeling/train.py",
              "metrics": {
                "accuracy":  "float - overall accuracy",
                "classification_report": "dict - per-class precision, recall, f1",
                "confusion_matrix": "numpy array",
                "predictions":  "numpy array of predicted labels",
                "true_labels": "numpy array of ground truth labels"
              },
              "lines": "317-323"
            }
          },
          {
            "id": 7,
            "question":  "What arguments does the explain_prediction_lime function accept? ",
            "answer": {
              "location": "XAI/modeling/predict.py",
              "arguments": {
                "model":  "PyTorch model",
                "image": "PIL. Image or numpy. ndarray",
                "transform": "Optional transform (default: None)",
                "num_samples": "Number of samples for LIME (default: 1000)",
                "save_path": "Path to save explanation (default: None)"
              },
              "lines": "236-249"
            }
          },
          {
            "id": 8,
            "question": "What feature extraction functions are called in extract_all_features? ",
            "answer": {
              "location": "XAI/features.py",
              "functions": [
                "extract_color_histogram()",
                "extract_shape_features()",
                "extract_texture_features()",
                "extract_lbp_features()",
                "extract_glcm_features()"
              ]
            }
          },
          {
            "id": 9,
            "question": "What hyperparameters does plot_tsne_features use for t-SNE? ",
            "answer": {
              "location": "XAI/plots.py",
              "hyperparameters": {
                "perplexity": 30,
                "n_iter": 1000,
                "n_components": 2,
                "random_state": 42
              },
              "lines": "323-360"
            }
          },
          {
            "id": 10,
            "question":  "What loss functions are supported in the main training function? ",
            "answer": {
              "location": "XAI/modeling/train.py",
              "loss_functions": {
                "binary":  "nn.BCEWithLogitsLoss()",
                "multiclass": "nn. CrossEntropyLoss()",
                "selection": "Based on isBinary flag"
              },
              "lines": "421"
            }
          }
        ]
      },
      "category_3_global_relational": {
        "description": "Cross-file flow queries - tracing functionality across multiple files",
        "questions": [
          {
            "id": 1,
            "question": "Trace the full data loading and preprocessing pipeline from download to DataLoader.",
            "answer": {
              "flow": [
                "1. download_and_extract_ham10000() downloads dataset via Kaggle API",
                "2. organize_data() sorts images by class into INTERIM_DATA_DIR",
                "3. prepare_data() loads metadata, optionally balances classes, splits into train/val/test",
                "4. Creates HAM10000Dataset instances with transforms from get_transforms()",
                "5. Returns PyTorch DataLoaders with configured batch size and workers"
              ],
              "files": ["XAI/dataset.py"],
              "lines":  "37-296"
            }
          },
          {
            "id":  2,
            "question": "Trace the complete model training flow from initialization to saving.",
            "answer": {
              "flow": [
                "1. main() calls set_seed() for reproducibility",
                "2. prepare_data() loads train/val/test data",
                "3. Model instantiated via ResizedModel(dl_models[i]. inputSize(), dl_models[i]())",
                "4. load_best_model() checks for existing checkpoint",
                "5. train_model() runs training loop with optimizer/scheduler",
                "6. evaluate_model() computes validation metrics each epoch",
                "7. Best model saved to MODELS_DIR with val_acc in filename",
                "8. test_model() evaluates final performance"
              ],
              "files": ["XAI/modeling/train.py", "XAI/modeling/AllModels.py", "XAI/modeling/ResizeLayer.py"]
            }
          },
          {
            "id": 3,
            "question": "How do the XAI explanation methods (LIME, SHAP, GradCAM) integrate with predictions?",
            "answer": {
              "flow": [
                "1. Model loaded via load_best_model()",
                "2. predict_image() generates base predictions",
                "3.  LIME:  explain_prediction_lime() uses lime_image. LimeImageExplainer with custom predict_fn wrapper",
                "4. SHAP:  explain_prediction_shap() uses shap.DeepExplainer with background tensor",
                "5. GradCAM:  explain_prediction_gcam() uses pytorch_grad_cam. GradCAM with target layers from get_target_layers()"
              ],
              "files":  ["XAI/modeling/predict.py", "XAI/explainability/"]
            }
          },
          {
            "id": 4,
            "question": "Trace how configuration values flow from config.py through the entire codebase.",
            "answer": {
              "config_file": "XAI/config.py",
              "config_values": {
                "paths": ["ROOT_DIR", "DATA_DIR", "MODELS_DIR", "FIGURES_DIR"],
                "model_params":  ["MODEL_INPUT_SIZE", "BATCH_SIZE", "NUM_EPOCHS", "LEARNING_RATE"],
                "class_mappings": ["CLASS_NAMES", "NUM_CLASSES"]
              },
              "consumers": {
                "dataset. py": "Data paths, batch size",
                "train.py": "Training params, model paths",
                "predict.py": "Class names, paths",
                "plots. py": "Class names, figure paths",
                "features. py": "Data paths",
                "model files": "NUM_CLASSES"
              }
            }
          },
          {
            "id": 5,
            "question": "How does the preprocessing pipeline connect to model input requirements?",
            "answer": {
              "flow": [
                "1. XAI/preprocessing/ contains HairRemoval, CLAHE, EnhanceClarityCV, ContrastStretch",
                "2. get_transforms() in dataset.py chains these with torchvision.transforms.v2",
                "3. HAM10000Dataset.__getitem__ applies transform pipeline",
                "4. ResizeLayer.py contains ResizedModel wrapper that resizes input to model's inputSize()",
                "5. Each model class defines inputSize() static method (e.g., 224x224)"
              ],
              "files": ["XAI/preprocessing/", "XAI/dataset.py", "XAI/modeling/ResizeLayer.py"]
            }
          },
          {
            "id":  6,
            "question": "Trace how feature extraction connects ML models to explainability.",
            "answer": {
              "flow": [
                "1. XAI/features. py defines extract_all_features() combining color, shape, texture, LBP, GLCM features",
                "2. Features saved to PROCESSED_DATA_DIR",
                "3. train_ml.py loads features, scales via StandardScaler, trains DTModel/RFModel",
                "4. XAI/explainability/shap_ml_explainer.py uses same features for SHAP explanations",
                "5. XAI/explainability/lime_ml_explainer.py uses features for LIME",
                "6. run_explainers. py orchestrates all ML explanations with analyze_image()"
              ],
              "files": ["XAI/features.py", "XAI/modeling/train_ml.py", "XAI/explainability/", "XAI/run_explainers. py"]
            }
          },
          {
            "id": 7,
            "question": "How do the different model architectures (CNN variants) share common base functionality?",
            "answer": {
              "architecture":  {
                "base_class": "Base_Model. py defines abstract BaseModel(nn.Module) with name() and inputSize() methods",
                "dl_models": ["SkinLesionCNN", "CustomCNN", "SkinCancerCNN", "D2CNN", "ModifiedInceptionV3", "FineTunedResNet50", "SkinEfficientNetB5", "MobileNetV2"],
                "collection": "AllModels.py collects all model classes in dl_models list",
                "usage": "train. py and predict.py iterate through dl_models for training/evaluation",
                "ml_hierarchy": "ML models use separate ML_Base_model.py hierarchy"
              }
            }
          },
          {
            "id": 8,
            "question": "Trace the complete explainability report generation flow in run_explainers.py.",
            "answer": {
              "flow": [
                "1. main() parses CLI args (image, model, features, output)",
                "2. analyze_image() loads model via load_latest_model(), scaler via load_feature_scaler()",
                "3. load_background_data() prepares reference data for explainers",
                "4. explain_image_with_feature_importance() generates feature rankings",
                "5. explain_prediction_with_lime() + visualize_lime_explanation()",
                "6. explain_prediction_with_shap() + visualize_shap_explanations()",
                "7. explain_top_features_for_image() for PDP analysis",
                "8. generate_combined_report() creates HTML summary with all visualizations"
              ],
              "file":  "XAI/run_explainers.py"
            }
          },
          {
            "id": 9,
            "question":  "How does checkpoint loading and resuming work across training runs?",
            "answer": {
              "flow": [
                "1. load_best_model() scans MODELS_DIR for {model_name}*. pth files",
                "2. Parses val_acc from filename pattern {name}-{val_acc}-e{epoch}-{timestamp}.pth",
                "3. Sorts by validation accuracy and returns best checkpoint",
                "4. main() loads model_state_dict and optimizer_state_dict from checkpoint",
                "5. Sets start_epoch = checkpoint['epoch'] + 1",
                "6. Sets best_val_acc = checkpoint['val_acc']",
                "7. train_model() continues from start_epoch with preserved best accuracy threshold"
              ],
              "file": "XAI/modeling/train.py",
              "lines":  "366-408, 436-477"
            }
          },
          {
            "id":  10,
            "question": "Trace how class balancing affects the entire pipeline from data to evaluation.",
            "answer": {
              "flow": [
                "1. prepare_data(balanced=True) in dataset.py",
                "2. Counts class distribution via metadata['dx'].value_counts()",
                "3. Finds min_class_count across all 7 classes",
                "4. Downsamples larger classes with class_df.sample(min_class_count)",
                "5. Balanced DataFrame used for train/val/test split with stratify=metadata['dx']",
                "6. HAM10000Dataset maps classes to indices via class_to_idx",
                "7. test_model() generates per-class metrics in classification_report",
                "8. plot_confusion_matrix() visualizes class-wise performance"
              ],
              "files": ["XAI/dataset.py", "XAI/modeling/train.py", "XAI/plots.py"]
            }
          }
        ]
      }
    }
  }
}
```

---

## Summary

This evaluation framework provides 30 questions across 3 complexity levels: 

- **Category 1 (Simple Lookup)**: Tests basic code navigation and definition finding
- **Category 2 (Local Context)**: Tests understanding of function internals, parameters, and local variables
- **Category 3 (Global/Relational)**: Tests understanding of cross-file dependencies, data flows, and architectural patterns

The questions are specifically tailored to the `Explainable-AI-for-Skin-Cancer-Detection` repository and cover key components including: 
- Configuration management
- Data loading and preprocessing
- Model architectures (Deep Learning & Machine Learning)
- Training and evaluation pipelines
- Explainability techniques (LIME, SHAP, GradCAM, PDP)
- Feature extraction and visualization
