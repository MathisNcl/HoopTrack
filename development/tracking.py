"""Tests"""
# import json
# from matplotlib import pyplot as plt
# from hooptrack.basketball.visualiser import Visualiser
# from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
# from PIL import Image

# import pandas as pd

# from hooptrack.basketball.utils import iou, calculate_center_distance
# from copy import deepcopy


# from dash import dcc, html, Dash
# from dash.dependencies import Input, Output, State
# import plotly.express as px


# with open("output_vid_2.json") as f:
#     r = json.load(f)


# res = Visualiser.track_labels(r)


# def plot_frame(frame_number, data):
#     plt.clf()  # Clear the previous plot
#     plt.figure(figsize=(8, 6))
#     for key, value in data.items():
#         sub_value = value[:frame_number]
#         if key == "ball":
#             plt.plot([c[0] for c in sub_value], [3840 - c[1] for c in sub_value], label=key)
#         else:
#             plt.plot(
#                 [c[0] for c in sub_value],
#                 [3840 - c[1] for c in sub_value],
#                 "o",
#                 label=key,
#             )
#     plt.title(f"Frame {frame_number}")
#     plt.xlabel("X-axis")
#     plt.ylabel("Y-axis")
#     plt.legend()
#     plt.grid(True)
#     plt.show()


# def plot_frame_export(frame_number, data):
#     fig, ax = plt.subplots(figsize=(8, 6))
#     for key, value in data.items():
#         x, y = zip(*value[:frame_number])
#         if key == "rim":
#             y = [3840 - coord for coord in y]  # Invert y-coordinate
#             ax.scatter(x, y, label=key)
#         elif key == "ball":
#             y = [3840 - coord for coord in y]  # Invert y-coordinate
#             ax.plot(x, y, label=key)
#     ax.set_title(f"Frame {frame_number}")
#     ax.set_xlabel("X-axis")
#     ax.set_ylabel("Y-axis")
#     ax.legend()
#     ax.grid(True)
#     canvas = FigureCanvas(fig)
#     canvas.draw()
#     plt.close()
#     return Image.frombytes("RGB", canvas.get_width_height(), canvas.tostring_rgb())


# # # Set the number of frames
# # num_frames = max(len(res["rim"]), len(res["ball"]))

# # # Generate frames and save as images
# # frames = [plot_frame_export(frame, res) for frame in range(1, num_frames + 1)]
# # frames[0].save(
# #     "animated_plot2.gif",
# #     format="GIF",
# #     append_images=frames[1:],
# #     save_all=True,
# #     duration=120,
# #     loop=0,
# # )


# def track_in_fine(frames, iou_thresh=0.5, distance_thresh=200):
#     """
#     Track the ball using IoU and center distance between detections.
#     """
#     tracked_object = {}

#     augmented_data = deepcopy(frames)

#     for idx, frame_data in frames.items():
#         if frame_data is None:
#             continue

#         current_boxes = [data["bbox"] for data in frame_data if data]
#         current_labels = [data["label_name"] for data in frame_data if data]

#         # j'ai récupéré toutes les bboxes de ballon

#         for i, curr_box in enumerate(current_boxes):
#             curr_label = current_labels[i]
#             if curr_label not in tracked_object:
#                 tracked_object[curr_label] = {}
#             if len(tracked_object[curr_label]) == 0:
#                 tracked_object[curr_label]["id_0"] = {
#                     "bboxes": [curr_box],
#                     "frames": [idx],
#                 }
#                 augmented_data[idx][i]["track_id"] = "id_0"
#                 continue

#             best_match_id, max_iou, distance_computed = tracking_best_match(
#                 tracked_object,
#                 curr_label,
#                 curr_box,
#                 int(idx),
#                 80,
#                 iou_thresh,
#                 distance_thresh,
#             )

#             if best_match_id is not None:
#                 tracked_object[curr_label][best_match_id]["bboxes"].append(curr_box)
#                 tracked_object[curr_label][best_match_id]["frames"].append(idx)
#                 augmented_data[idx][i]["track_id"] = best_match_id
#             else:
#                 tracked_object[curr_label][f"id_{len(tracked_object[curr_label])}"] = {
#                     "bboxes": [curr_box],
#                     "frames": [idx],
#                 }

#                 augmented_data[idx][i]["track_id"] = f"id_{len(tracked_object[curr_label])}"

#     return tracked_object, augmented_data


# tracking, augmented_data = track_in_fine(r, iou_thresh=0.5, distance_thresh=400)
# print(len(tracking["ball"]))
# print({k: len(v["frames"]) for k, v in tracking["ball"].items()})


# def track_to_df(augmented_data):
#     df_dict = {
#         "frame_id": [],
#         "bbox": [],
#         "label_name": [],
#         "label_id": [],
#         "score": [],
#         "track_id": [],
#         "center_x": [],
#         "center_y": [],
#     }
#     for frame_id, objects in augmented_data.items():
#         if objects is None or len(objects) == 0:
#             for _, v in df_dict.items():
#                 if v == "frame_id":
#                     v.append(int(frame_id))
#                 else:
#                     v.append(None)
#             continue
#         #
#         for object in objects:
#             df_dict["frame_id"].append(int(frame_id))
#             x1, y1, x2, y2 = object["bbox"]
#             df_dict["center_x"].append(int((x1 + x2) / 2))
#             df_dict["center_y"].append(2160 - int((y1 + y2) / 2))
#             for k, v in object.items():
#                 df_dict[k].append(v)

#     df = pd.DataFrame(df_dict)
#     df["track_id_label"] = df["label_name"] + df["track_id"]
#     return df


# df = track_to_df(augmented_data)

# external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

# app = Dash(__name__, external_stylesheets=external_stylesheets)


# def make_fig(selected_frame):
#     filtered_df = df[df["frame_id"] <= selected_frame]
#     fig = px.line(
#         filtered_df,
#         x="center_x",
#         y="center_y",
#         color="track_id_label",
#         markers=True,
#         labels={"center_x": "Center X", "center_y": "Center Y"},
#         title="Center Point Animation with Slider",
#         hover_data=["track_id_label"],
#     )
#     fig.update_layout(
#         xaxis=dict(range=[0, 3840]),
#         yaxis=dict(range=[0, 2160]),
#     )
#     return fig


# app.layout = html.Div(
#     [
#         dcc.Interval(id="animate", disabled=True),
#         dcc.Graph(id="graph-with-slider", figure=make_fig(0)),
#         dcc.Slider(
#             id="frame-slider",
#             min=df["frame_id"].min(),
#             max=df["frame_id"].max(),
#             value=df["frame_id"].min(),
#             step=1,
#         ),
#         html.Button("Play", id="play"),
#     ]
# )


# @app.callback(
#     Output("graph-with-slider", "figure"),
#     Output("frame-slider", "value"),
#     Input("animate", "n_intervals"),
#     Input("frame-slider", "value"),
#     prevent_initial_call=True,
# )
# def update_figure(n, selected_frame):
#     return make_fig(selected_frame), n * 4


# @app.callback(
#     Output("animate", "disabled"),
#     Input("play", "n_clicks"),
#     State("animate", "disabled"),
# )
# def toggle(n, playing):
#     if n:
#         return not playing
#     return playing


# # Run the app
# if __name__ == "__main__":
#     app.run_server(debug=True)


# import numpy as np


# def create_cost_matrix(boxes1, boxes2):
#     cost_matrix = np.zeros((len(boxes1), len(boxes2)))
#     for i, box1 in enumerate(boxes1):
#         for j, box2 in enumerate(boxes2):
#             cost_matrix[i, j] = 1 - iou(box1, box2)
#     return cost_matrix


# b1 = [b["bbox"] for b in r["243"]]
# b2 = [b["bbox"] for b in r["244"]]

# create_cost_matrix(b1, b2)


# def calculate_color_similarity(box1, box2, image1, image2):
#     # Calculer la similarité d'apparence basée sur la différence de couleur moyenne
#     box1 = [0 if b < 0 else b for b in box1]
#     box2 = [0 if b < 0 else b for b in box2]
#     mean_color_box1 = np.mean(image1[int(box1[1]) : int(box1[3]), int(box1[0]) : int(box1[2])], axis=(0, 1))
#     mean_color_box2 = np.mean(image2[int(box2[1]) : int(box2[3]), int(box2[0]) : int(box2[2])], axis=(0, 1))
#     color_difference = np.linalg.norm(mean_color_box1 - mean_color_box2)

#     return 1 / (1 + np.exp(-color_difference))


# def create_cost_matrix(boxes1, boxes2, image1, image2):
#     # Créer la matrice des coûts basée sur l'IoU et la similarité d'apparence
#     cost_matrix = np.zeros((len(boxes1), len(boxes2)))
#     for i, box1 in enumerate(boxes1):
#         for j, box2 in enumerate(boxes2):
#             iou_cost = 1 - iou(box1, box2)
#             color_similarity_cost = calculate_color_similarity(box1, box2, image1, image2)

#             cost_matrix[i, j] = 0.65 * iou_cost + 0.35 * color_similarity_cost
#     return cost_matrix


# from scipy.optimize import linear_sum_assignment


# def object_tracking(previous_boxes, current_boxes, previous_image, current_image, threshold=0.5):
#     cost_matrix = create_cost_matrix(previous_boxes, current_boxes, previous_image, current_image)

#     row_ind, col_ind = linear_sum_assignment(cost_matrix)

#     is_association = [cost_matrix[i, j] < threshold for i, j in zip(row_ind, col_ind)]

#     # for i, is_associated in zip(col_ind, is_association):
#     #     current_boxes[i]

#     return is_association


# im1 = Image.open("243.jpg")
# im2 = Image.open("244.jpg")
# create_cost_matrix(
#     b1,
#     b2,
#     np.array(im1),
#     np.array(im2),
# )


# object_tracking(
#     b1,
#     b2,
#     np.array(im1),
#     np.array(im2),
# )
