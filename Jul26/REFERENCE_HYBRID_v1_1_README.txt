T-CORE+OTHER FINAL HYBRID 4B + UAVVASTE STAGE-A — v1.1
======================================================================

IMPORTANT v1.1 FIXES
--------------------
This version fixes the compare-pipelines failure:

    ERROR: name 'load' is not defined

The error was inside:

    10_test_evaluate_render.py

The script called load() and fmt() but the helper definitions were omitted.
Both helpers are restored in v1.1.

The pipeline comparison can now be launched with:

    .\RUN_COMPARE_PIPELINES.ps1

or:

    .\RUN_HYBRID_FINAL.ps1 `
        -Stage compare-pipelines `
        -Overwrite


STANDALONE APP — NOW PROMINENT
------------------------------
The standalone GUI is now a top-level file:

    STANDALONE_APP.py

Recommended launcher:

    .\START_STANDALONE_APP.ps1

The previous launcher also works:

    .\RUN_VISUAL_APP.ps1

or through the unified runner:

    .\RUN_HYBRID_FINAL.ps1 -Stage app


CCBRE AI-LAB BRANDING
---------------------
The supplied CCBRE AI-LAB logo is included at:

    assets\CCBRE_AI_LAB_Logo.png
    assets\CCBRE_AI_LAB_Logo.ico

It is displayed in the standalone GUI and is also used as the Windows
application icon when BUILD_WINDOWS_APP.ps1 is run.


OPTIONAL MODEL SELECTION IN THE APP
-----------------------------------
The app starts with:

    Hybrid FINAL_MODELS (recommended)

which uses:

    FINAL_MODELS\StageA_AnyFM_Detector\best.pt
    FINAL_MODELS\StageB_Binary_Verifier\best.pt
    FINAL_MODELS\StageC_Exact_Classifier\best.pt

The GUI can also select:

    Base 4B BASE_FINAL_MODELS

to run the original base three-stage pipeline.

A third option is:

    Custom checkpoints

For custom mode, Stage A, Stage B, and Stage C can each be selected separately
with a Browse button.

This is useful for:

    comparing original Stage A versus UAVVaste Stage A;
    testing alternative Stage-A candidates;
    testing a different Stage-B verifier without changing FINAL_MODELS;
    testing a different Stage-C exact classifier;
    engineering/debugging experiments.

The selected model paths and operating point are saved into every output JSON
under:

    models_used

The app also provides:

    Validate selected models

before inference.


APP GT BEHAVIOR
---------------
The automatic behavior remains unchanged:

    matching YOLO .txt exists
        -> GT comparison automatically
        -> GT + predictions together
        -> TP / FP / FN
        -> diagnostic category

    no matching YOLO .txt
        -> normal inference automatically
        -> no GT-based claim

Diagnostic categories:

    correct_negative
    perfect_detection
    false_positive_only
    false_negative_only
    mixed_error

Long labels remain in the readable top panel; only compact G1/G2/P1/P2 labels
are placed over the image.


QUICK COMMANDS AFTER v1.1 UPDATE
--------------------------------
1. Verify the package code:

    .\CHECK_PACKAGE.ps1

Expected:

    "passed": true


2. Compare complete base versus hybrid pipelines:

    .\RUN_COMPARE_PIPELINES.ps1


3. Run the final hybrid target test:

    .\RUN_FINAL_TEST.ps1


4. Start standalone app:

    .\START_STANDALONE_APP.ps1


WINDOWS STANDALONE BUILD
------------------------
Run:

    .\BUILD_WINDOWS_APP.ps1

Output:

    dist\CCBRE_AI_LAB_Foreign_Material_Detection\

The build includes:

    CCBRE logo
    CCBRE app icon
    config.json
    Hybrid/Base/Custom model selection GUI


MODEL-SELECTION SAFETY
----------------------
Changing the model selection in the app does NOT copy, overwrite, fine-tune, or
modify any model.

It only changes which checkpoints are loaded for the current inference session.

Hybrid FINAL_MODELS remains the recommended default.

======================================================================
ORIGINAL v1.0 PRODUCT DOCUMENTATION FOLLOWS
======================================================================

T-CORE+OTHER FINAL HYBRID PRODUCT — 4B + UAVVASTE STAGE-A v1.0
====================================================================

PRODUCT DECISION
----------------
This package is the deployable follow-on to the final three-stage 4B product.

The measured target-validation experiment showed:

    Stage A: UAVVaste fine-tuning IMPROVED detection/localization performance.
    Stage B: UAVVaste fine-tuning DEGRADED the binary verifier.

Therefore the final hybrid architecture is deliberately:

    Stage A = target-trained 4B Stage A + controlled UAVVaste fine-tuning
    Stage B = target-trained/base 4B Stage B, unchanged
    Stage C = target-trained/base 4B Stage C, unchanged

The observed comparison is included under:

    REFERENCE_RESULTS\observed_uavvaste_comparison.json
    REFERENCE_RESULTS\SCIENTIFIC_DECISION.json

OBSERVED STAGE-A IMPROVEMENT
----------------------------
On TARGET validation:

    precision   0.839607 -> 0.841616
    recall      0.668785 -> 0.711356     (+0.042572)
    mAP50       0.748696 -> 0.780215     (+0.031518)
    mAP50-95    0.588829 -> 0.610472     (+0.021644)

Stage B was therefore NOT fine-tuned in this product.

FOLDER / MODEL DESIGN
---------------------
This product uses a separate run root and does not overwrite the existing 4B run:

    C:\Hadi\Foreign_Material\Runs\T_Core_Other_T400_Hybrid_UAVVaste\

Base target models:

    BASE_FINAL_MODELS\
        StageA_AnyFM_Detector\best.pt
        StageB_Binary_Verifier\best.pt
        StageC_Exact_Classifier\best.pt

UAVVaste Stage-A training run:

    UAVVASTE_FINETUNE\STAGEA\
        StageA_4B_Finetuned_TargetPlusUAVVaste\weights\best.pt

Deployable hybrid models:

    FINAL_MODELS\
        StageA_AnyFM_Detector\best.pt       <- UAVVaste-fine-tuned if accepted
        StageB_Binary_Verifier\best.pt      <- base 4B, unchanged
        StageC_Exact_Classifier\best.pt     <- base 4B, unchanged
        final_model_manifest.json

TWO WAYS TO BUILD THE PRODUCT
-----------------------------
A. RECOMMENDED NOW — use the final 4B models already available:

    .\RUN_HYBRID_FINAL.ps1 -Stage check
    .\RUN_HYBRID_FINAL.ps1 -Stage import-base
    .\RUN_HYBRID_FINAL.ps1 -Stage prepare-uavvaste -Download
    .\RUN_HYBRID_FINAL.ps1 -Stage verify-uavvaste
    .\RUN_HYBRID_FINAL.ps1 -Stage finetune-stagea
    .\RUN_HYBRID_FINAL.ps1 -Stage compare-stagea
    .\RUN_HYBRID_FINAL.ps1 -Stage consolidate
    .\RUN_HYBRID_FINAL.ps1 -Stage verify-final
    .\RUN_HYBRID_FINAL.ps1 -Stage compare-pipelines
    .\RUN_HYBRID_FINAL.ps1 -Stage test

Or execute the same workflow automatically:

    .\RUN_HYBRID_FINAL.ps1 -Stage all-existing -Download

B. FULL REPRODUCIBILITY — rebuild from the project data first:

    .\RUN_HYBRID_FINAL.ps1 -Stage create
    .\RUN_HYBRID_FINAL.ps1 -Stage verify
    .\RUN_HYBRID_FINAL.ps1 -Stage train-base
    .\RUN_HYBRID_FINAL.ps1 -Stage prepare-uavvaste -Download
    .\RUN_HYBRID_FINAL.ps1 -Stage verify-uavvaste
    .\RUN_HYBRID_FINAL.ps1 -Stage finetune-stagea
    .\RUN_HYBRID_FINAL.ps1 -Stage compare-stagea
    .\RUN_HYBRID_FINAL.ps1 -Stage consolidate
    .\RUN_HYBRID_FINAL.ps1 -Stage verify-final
    .\RUN_HYBRID_FINAL.ps1 -Stage compare-pipelines
    .\RUN_HYBRID_FINAL.ps1 -Stage test

Or:

    .\RUN_HYBRID_FINAL.ps1 -Stage all-from-data -Download

DATA CREATION
-------------
01_create_dataset.py reproduces the configurable T-Core+Other dataset.

At T=400 the operational taxonomy is:

    0 BAW-Sack
    1 Papier
    2 Plastiksack
    3 Other

02_verify_dataset.py checks labels/images/splits and renders verification examples.

BASE MODEL TRAINING — TARGET DATA ONLY
--------------------------------------
03_train_base_models.py reproduces the final three-stage target-data workflow:

    Stage A: one-class Any-FM detector
    Stage B: ForeignMaterial / Background_NotFO verifier
    Stage C: BAW-Sack / Papier / Plastiksack / Other exact classifier

It generates:

    TRAINING_ASSETS\StageA_Final_Replay_View
    TRAINING_ASSETS\StageBC_Crop_Pool
    TRAINING_ASSETS\StageBC_Final_Views
    BASE_FINAL_MODELS\...

For the current computer, import-base is faster because the final 4B models are
already available at:

    C:\Hadi\Foreign_Material\Runs\T_Core_Other_T400_Final\FINAL_MODELS

UAVVASTE
--------
05_prepare_uavvaste.py:

    downloads/checks the official archive when requested;
    validates the published MD5;
    converts COCO rubbish bounding boxes to one-class YOLO ForeignMaterial;
    keeps official train/val/test provenance;
    also preserves converted polygons for future research.

06_verify_uavvaste.py:

    checks 540/116/116 official splits;
    checks one-class label coordinates;
    counts images/objects;
    renders random annotation examples.

Only official UAVVaste TRAIN is used in Stage-A fine-tuning.
UAVVaste val/test never participate in target model selection.

STAGE-A FINE-TUNING ONLY
------------------------
07_finetune_stageA_uavvaste.py automatically builds:

    TRAINING_ASSETS\StageA_TargetPlusUAVVaste

The train split contains:

    original target Stage-A training view
    + controlled UAVVaste official-train images

The validation split is copied from the TARGET Stage-A validation only.

Default external exposure:

    min(540, 30% of target Stage-A train image count)

Fine-tuning begins from:

    BASE_FINAL_MODELS\StageA_AnyFM_Detector\best.pt

Stage B and Stage C are not touched.

AMP remains enabled for efficiency, but the package suppresses the redundant
Ultralytics auxiliary AMP-model check that previously downloaded yolo26n.pt.
The actual initializer is always printed before training.

STAGE-A ACCEPTANCE
------------------
08_compare_stageA.py compares base and fine-tuned Stage A on TARGET validation.

Default acceptance requires:

    recall gain >= 0.01
    mAP50 gain >= 0.00
    mAP50-95 gain >= 0.00
    precision drop no worse than -0.01

The result is saved to:

    UAVVASTE_FINETUNE\COMPARISON\stageA_comparison.json

09_consolidate_hybrid_models.py uses the fine-tuned A only when this target-domain
acceptance passes. B and C always come from BASE_FINAL_MODELS.

13_verify_final_product.py verifies SHA-256 provenance and specifically requires
Stage B and Stage C in FINAL_MODELS to be byte-identical to BASE_FINAL_MODELS.
Run:

    .\RUN_HYBRID_FINAL.ps1 -Stage verify-final

Use --load-models directly with the Python verifier for an optional Ultralytics
checkpoint-load smoke test.

COMPLETE THREE-STAGE TESTING
----------------------------
10_test_evaluate_render.py is the mature Phase-4B evaluator with:

    Any-FO image confusion matrix
    object confusion matrix
    per-class metrics
    same-class micro/macro F1
    class-aware F1
    geometric localization recall
    Stage-A native metrics
    GT + predictions together
    readable top information panel
    compact G1/G2/P1/P2 labels on the image

Rendered images are organized as:

    correct_negative
    perfect_detection
    false_positive_only
    false_negative_only
    mixed_error

11_compare_base_vs_hybrid.py evaluates BASE_FINAL_MODELS and FINAL_MODELS on the
same TARGET validation split and writes:

    PIPELINE_COMPARISON\comparison_val.json

This is important because the final question is not only whether Stage A improves,
but whether the improved proposals help the entire A+B+C system while the original
Stage B rejects extra false proposals.

STANDALONE APP
--------------
Run:

    .\RUN_VISUAL_APP.ps1

or:

    .\RUN_HYBRID_FINAL.ps1 -Stage app

The app automatically behaves as follows:

    matching YOLO .txt exists
        -> GT comparison
        -> GT + predictions
        -> TP / FP / FN
        -> diagnostic category

    no matching YOLO .txt
        -> normal inference only
        -> no unsupported GT-based category

The information panel is above the original photo. Long text does not obscure
objects. The image itself contains compact G/P IDs.

APP outputs:

    APP_OUTPUTS\UNLABELED_INFERENCE\
    APP_OUTPUTS\GT_COMPARISON\<category>\

Each analysis saves rendered JPG + JSON.

WINDOWS APP
-----------
BUILD_WINDOWS_APP.ps1 remains included for creating a Windows standalone build.

MODEL PROVENANCE
----------------
FINAL_MODELS\final_model_manifest.json records:

    exact source of A/B/C
    target Stage-A validation decision
    SHA-256 hashes
    operating point

This makes it explicit that:

    A = UAVVaste-improved target detector
    B = target-only 4B binary verifier
    C = target-only 4B exact classifier

NO STAGE-B UAVVASTE FINE-TUNING
-------------------------------
The earlier experiment is retained only as scientific evidence under
REFERENCE_RESULTS. It is not part of training or deployment because it degraded:

    target accuracy
    FO recall
    background specificity
    balanced accuracy
    FO F1

This exclusion is intentional, not an omission.
