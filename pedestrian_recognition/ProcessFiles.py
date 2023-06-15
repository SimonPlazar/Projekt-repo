import DetectOnFile
import os

video_folder = os.path.join("no_audio", "Testiranje")
for video_name in os.listdir(video_folder):
    print(f"Working on {video_name}..\n")
    video_path = os.path.join(video_folder, video_name)
    output_video_path = os.path.join("Output", "Testiranje", "output_" + video_name)

    DetectOnFile.Detect(video_path, output_video_path)
    print(f"Done with {video_name}..\n")