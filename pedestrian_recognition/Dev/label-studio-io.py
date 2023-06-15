import requests
from label_studio_sdk import Client

# prepoznava pešca z že naprej naretim modelom
# diagram poteka zaznave pešcca -> repsonse modela

# pasivni sistemi
# en za notranjo uporabo -> enej
# en za zunanjo -> zaznava pešca

label_studio_url = "http://localhost:8080"
access_token = "6251cf4d2b9d967c4b3b8f2b0fff307282f93c93"

# Vzpostavitev povezave z label studio
client = Client(label_studio_url, access_token)

Projekt = client.get_project(1) # id = 1
tasks = Projekt.get_tasks()

for task in tasks:
    annotations = task.get("annotations")
    if annotations:
        result = annotations[0].get("result")
        if result:
            video_url = task["data"]["video"]
            print(f"Working on: {video_url}")
            video_file = requests.get(label_studio_url + video_url, headers={"Authorization": f"Token {access_token}"})

            video_split = []
            for annotation in result:
                value = annotation["value"]
                sequence = value["sequence"]
                # filter out the frames where 'Human' is detected
                human_detected_frames = [frame for frame in sequence if
                                         frame["enabled"] and 'Human' in value.get('labels', [])]
                if human_detected_frames:
                    # get the first and last frame where 'Human' is detected
                    start_time = human_detected_frames[0]["time"]
                    end_time = human_detected_frames[-1]["time"]

                    # convert time in seconds to mm:ss:ms
                    start_time_str = f"{int(start_time // 60):02d}:{int(start_time % 60):02d}:{int((start_time % 1) * 1000):03d}"
                    end_time_str = f"{int(end_time // 60):02d}:{int(end_time % 60):02d}:{int((end_time % 1) * 1000):03d}"

                    print(f'Start time: {start_time_str}, End time: {end_time_str}')


