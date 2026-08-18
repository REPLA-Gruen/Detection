CCBRE AI-LAB — FINAL CONTEXT + FOREIGN-MATERIAL PRODUCT v1.0
================================================================
Status: final integrated engineering product before Phase 5.

ARCHITECTURE
------------
Stage 0 — D1 context classification:
    Container / Wagon + confidence

Stage A — Any-Foreign-Material detector:
    target-trained Phase-4B detector, then UAVVaste fine-tuning only when the
    target-validation acceptance rule is passed.

Stage B — binary verifier:
    ForeignMaterial / Background_NotFO; target-trained only.

Stage C — exact class:
    BAW-Sack / Papier / Plastiksack / Other; target-trained only.

IMPORTANT: Stage 0 is metadata in this final pre-Phase-5 product. It is run on
all images and stored/rendered, but it does NOT gate or suppress A/B/C. That
avoids introducing a new routing dependency before it is scientifically tested.

ONE PRODUCT ROOT
----------------
Default run root:

C:\Hadi\Foreign_Material\Runs\CCBRE_ContextHybrid_T400_Final

Final deployable checkpoints:

FINAL_MODELS\
  Stage0_Context_Classifier\best.pt
  Stage0_Context_Reference_YOLOv8m\best.pt   [when available; optional]
  StageA_AnyFM_Detector\best.pt
  StageB_Binary_Verifier\best.pt
  StageC_Exact_Classifier\best.pt
  final_model_manifest.json
  FINAL_PRODUCT_VERIFICATION.json

D1 DATASET POLICY
-----------------
D1 is NOT randomly split again.

20_create_D1_aligned.py reads the T-Core image_manifest.csv and preserves the
same train / val / test membership for context classification. This makes the
context and foreign-material components refer to the same master partitions.

D1 classes remain filename-derived acquisition context:
    ...w        -> wagon
    ...c/b/t    -> container

Unknown context rows are skipped and reported.

D1 DEFAULT MODEL
----------------
The deployment priority is YOLO11m-cls, consistent with the existing D1
engineering recommendation: both models performed near-perfectly, while
YOLO11m offered a materially smaller checkpoint/parameter footprint.

The unified workflow can also train/retain YOLOv8m-cls as a reference model.
Selection is based on VALIDATION only with a minimum validation-accuracy floor.
The test split is never used to switch the selected context model.

NORMAL WORKFLOW — USING YOUR ALREADY TRAINED CURRENT PRODUCTS
-------------------------------------------------------------
Activate environment and enter this package folder:

C:\Hadi\venvs\fm-yolo11\Scripts\Activate.ps1
cd <this extracted folder>

1. Check paths:

.\RUN_FINAL_PRODUCT.ps1 -Stage check

2. Create D1 aligned to the existing T-Core split:

.\RUN_FINAL_PRODUCT.ps1 -Stage create-context -Overwrite
.\RUN_FINAL_PRODUCT.ps1 -Stage verify-context

3. Import existing D1 models:

.\RUN_FINAL_PRODUCT.ps1 -Stage import-context -Overwrite

4. Evaluate/select D1 context model from aligned validation:

.\RUN_FINAL_PRODUCT.ps1 -Stage evaluate-context -Overwrite

NOTE: for legacy imported D1 checkpoints, the aligned test result is an
engineering diagnostic unless its membership is proven untouched by the old
training split. The script writes this limitation into its decision JSON.

5. Import the CURRENT final hybrid A/B/C product:

.\RUN_FINAL_PRODUCT.ps1 -Stage import-foreign -Overwrite

6. Consolidate one four-stage FINAL_MODELS:

.\RUN_FINAL_PRODUCT.ps1 -Stage consolidate -Overwrite

7. Verify model hashes/roles/provenance:

.\RUN_FINAL_PRODUCT.ps1 -Stage verify-final

Optional stronger verification that actually loads all four models:

python .\27_verify_unified_product.py --load-models

8. Complete test command:

.\RUN_FINAL_PRODUCT.ps1 -Stage test-complete -Overwrite

This runs:
    D1 context test
    complete A/B/C foreign-material test with GT-aware categorized rendering
    combined COMPLETE_SYSTEM_SUMMARY.json

9. Start the ONLY standalone app in this package:

.\START_APP.ps1

The entire existing-product path can also be run with:

.\RUN_FINAL_PRODUCT.ps1 -Stage all-existing -Overwrite

FROM SCRATCH TO END
-------------------
The full reproducible route is:

.\RUN_FINAL_PRODUCT.ps1 -Stage create-target -Overwrite
.\RUN_FINAL_PRODUCT.ps1 -Stage verify-target
.\RUN_FINAL_PRODUCT.ps1 -Stage create-context -Overwrite
.\RUN_FINAL_PRODUCT.ps1 -Stage verify-context
.\RUN_FINAL_PRODUCT.ps1 -Stage train-context -Overwrite
.\RUN_FINAL_PRODUCT.ps1 -Stage evaluate-context -Overwrite
.\RUN_FINAL_PRODUCT.ps1 -Stage train-foreign-base -Overwrite

This clean rebuild uses generic Ultralytics pretrained weights (YOLOv8m / YOLOv8m-cls), NOT any prior project Phase-4/4B checkpoints.
.\RUN_FINAL_PRODUCT.ps1 -Stage prepare-uavvaste -Download
.\RUN_FINAL_PRODUCT.ps1 -Stage verify-uavvaste
.\RUN_FINAL_PRODUCT.ps1 -Stage finetune-stagea -Overwrite
.\RUN_FINAL_PRODUCT.ps1 -Stage compare-stagea
.\RUN_FINAL_PRODUCT.ps1 -Stage consolidate -Overwrite
.\RUN_FINAL_PRODUCT.ps1 -Stage verify-final
.\RUN_FINAL_PRODUCT.ps1 -Stage compare-foreign -Overwrite
.\RUN_FINAL_PRODUCT.ps1 -Stage test-complete -Overwrite

Or in one command:

.\RUN_FINAL_PRODUCT.ps1 -Stage all-from-data -Download -Overwrite

UAVVASTE POLICY
---------------
Only Stage A is fine-tuned with UAVVaste.
Stage B and Stage C remain target-trained models.

The Stage-A candidate is accepted only when target validation satisfies the
configured recall/mAP/precision guardrails. UAVVaste validation/test images are
not used to select the target-domain operating point.

STANDALONE APP — COMPACT UI
---------------------------
The new GUI intentionally removes all permanent model-path, operating-point,
and advanced-option rows from the main window.

Visible by default:
    compact CCBRE logo/title bar
    large image-analysis canvas
    resizable analysis-result pane
    small status bar

Advanced controls are hidden under menus:
    Models -> custom Stage 0 / A / B / C checkpoints
    Options -> operating point
    View -> show/hide analysis pane, saved-render information panel, fullscreen
    Tools -> model validation / output folder

Opening an image automatically starts analysis. Re-analysis is available from File.

Keyboard:
    Ctrl+O  open image and analyze
    Ctrl+R  re-analyze current image
    Ctrl+S  save rendered result
    F9      show/hide analysis text pane
    F11     fullscreen

The viewer displays the annotated PHOTO only by default. The long diagnostic
panel is still saved in the rendered JPG and can optionally be shown in the
viewer from View -> Show saved-render information panel in viewer.

GT BEHAVIOR
-----------
Foreign-material GT:
    matching YOLO .txt found -> automatic GT comparison and categories
    no .txt -> normal unlabeled inference

Context GT:
    automatically read when the image is inside a container/ or wagon/ D1
    class folder. Filename-suffix context GT is OFF by default and can be
    enabled under Options only for project audit use.

OUTPUT JSON
-----------
Every app analysis saves:
    context class/confidence/probabilities
    context checkpoint
    foreign-material alarm
    every verified box and exact/uncertain class
    Det / Ver / Cls scores
    GT comparison when available
    model paths and operating point
    rendered-image path

BATCH ANALYSIS
--------------
Example:

python .\29_batch_analyze_folder.py `
  --input C:\path\to\images `
  --output C:\path\to\analysis `
  --recursive

WINDOWS APP
-----------
Build the branded Windows application:

.\BUILD_WINDOWS_APP.ps1

Package output:

dist\CCBRE_AI_LAB_Context_Foreign_Material\

SCIENTIFIC EVIDENCE BOUNDARY
----------------------------
D1: the existing report found near-perfect validation performance but also
identified that its historical 3,359-image "test" evaluation covered the full
corpus and was not an independent test. The unified from-scratch workflow fixes
this implementation issue by preserving an explicit aligned 336-image D1 test
folder and evaluating it only after validation-based model selection.

For legacy imported D1 models, historical split overlap must still be treated
cautiously unless the old split manifest is proven identical/independent.

Foreign material: the current hybrid remains an engineering candidate. The
previous Phase-4B locked validation limitation is not erased by packaging.
UAVVaste provides target-validation evidence for Stage-A transfer learning,
not a new independent-test claim.

Phase 5 should start only after this four-stage package is operationally frozen.
