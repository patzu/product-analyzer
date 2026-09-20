# Labeling and model evaluation

Start with a written product taxonomy and concrete acceptance criteria. Decide
whether each image contains one product or many, which defects are visible, and
when occlusion or poor lighting requires UNKNOWN. Have ambiguous examples
reviewed rather than forcing a good/defective label.

For detection, use an annotation tool that exports bounding boxes and a stable
class mapping. YOLO labels generally use one text file per image with class ID
and normalized center-x, center-y, width, height. Keep class IDs consistent and
inspect overlays before any training. The image-only split utility does not
move those labels automatically.

For classification, label object crops and document resize, RGB conversion,
normalization, and class order. The optional TorchScript adapter has a specific
RGB [0,1] NCHW input contract. Models trained with different normalization need
an appropriate adapter or preprocessing exported inside the model.

No training pipeline or model weights are supplied. Use independent capture
sessions for train/validation/test, select thresholds on validation, and report
final metrics on the untouched test set. Compare any learned approach with the
classical baseline and include failure examples and latency measurements.
