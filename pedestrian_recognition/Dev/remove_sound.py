from moviepy.editor import VideoFileClip
import os

def remove_sound(file_name):
    posnetki_path = "Posnetki"

    video_path = os.path.join(posnetki_path, file_name)
    output_path = os.path.join("../no_audio", file_name)

    videoclip = VideoFileClip(video_path)
    new_clip = videoclip.without_audio()
    new_clip.write_videofile(output_path)

if __name__ == "__main__":

    # for file in os.listdir("Posnetki"):
    #     if file.endswith(".mp4"):
    #         remove_sound(file)

    remove_sound("meritev_1684078148_005.mp4")