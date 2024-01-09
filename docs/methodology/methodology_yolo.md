# Methodology basketball detection using yolo

After many tries, here are informations about how the model has been trained:

- Model: [Yolov8](https://docs.ultralytics.com/modes/train/#key-features-of-train-mode)
- Dataset: [RoboFlow dataset](https://universe.roboflow.com/ownprojects/basketball-w2xcw)
- Parameters (others are default one):
  - Initial model: yolov8n.pt
  - Epochs: 100
  - Batch size: 64

???+ note
    Dataset used had been divided by two to save time

## Metrics on validation

Accuracy     | Recall        | mAP 50       | mAP 50-95
------------ | ------------- | ------------ | ------------
Content Cell | Content Cell  | Content Cell | Content Cell

## Post processing

In order to ensure reliability, probability threshold for detection is set to 0.25 and some checks are performed.
A sliding window of 15 images is used to filter wrong detction - with 30fps, it is 0.5 second.

Insérer une formule avec deux lignes : 1 si distance moyenne sur les 15 dernières images supérieure à threshold  0 sinon
avec threshold = x

$$
\text{condition} = \begin{cases}
1 & \text{si } \frac{1}{15} \sum_{k=1}^{15} \sqrt{(x_k - y_k)^2} < th \\
0 & \text{sinon, où} \\
x_k \text{and} \ y_k & \ \text{basketball bounding box center} \\
th & \text{threshold distance} = 10
\end{cases}
$$

TODO
