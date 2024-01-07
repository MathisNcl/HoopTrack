"""Script test"""
# from hooptrack.basketball.inference_pipeline import InferencePipelineBasketBall
# from hooptrack.schemas.config import BasketballDetectionConfig

# from hooptrack.basketball.visualiser import Visualiser
# from PIL import Image
# from cProfile import Profile
# from pstats import SortKey, Stats


# with Profile() as profile:
#     inference_pipeline = InferencePipelineBasketBall(
#         config=BasketballDetectionConfig(
#             model="/Users/mathisnicoli/Desktop/PROJETS/HoopTrack/models/best_nano_22h21.onnx"
#         )
#     )
#     res = inference_pipeline.run("/Users/mathisnicoli/Desktop/PROJETS/HoopTrack/data/img/capture1.png")
#     (Stats(profile).strip_dirs().sort_stats(SortKey.TIME).print_stats())
# visualiser = Visualiser()

# im = visualiser.plot(res, show_id=True)

# Image.fromarray(im).show()


from hooptrack.basketball.inference_pipeline import InferencePipelineBasketBall
from hooptrack.schemas.config import BasketballDetectionConfig

inference_pipeline = InferencePipelineBasketBall(
    config=BasketballDetectionConfig(
        model="/Users/mathisnicoli/Desktop/PROJETS/HoopTrack/models/best_nano_22h21.onnx", frame_processed_every=6
    )
)

inference_pipeline.live_streaming(source="/Users/mathisnicoli/Downloads/IMG_6226.MOV", save=True, show=False)


inference_pipeline.live_streaming(source=0, save=True, show=False)
