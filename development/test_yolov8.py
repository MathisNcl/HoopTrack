"""Script test"""
# from ultralytics import YOLO

# stream = True to save lot of time
# model = YOLO("models/best_nano_22h21.pt")
# res = model.predict(
#     "data/VID1.mp4",
#     save=True,
#     conf=0.25,
#     max_det=4,
#     save_txt=True,
# )
# model.predict(source="0", show=True)
# model.export(format="onnx")


from hooptrack.basketball.inference_pipeline import InferencePipelineBasketBall
from hooptrack.schemas.config import BasketballDetectionConfig

inference_pipeline = InferencePipelineBasketBall(
    config=BasketballDetectionConfig(model="/Users/mathisnicoli/Desktop/PROJETS/HoopTrack/models/best_nano_22h21.onnx")
)

res = inference_pipeline.run("/Users/mathisnicoli/Desktop/PROJETS/HoopTrack/data/img/capture1.png")

print(res)
