import getopt
import sys
import DetectOnFile
import os

argv = sys.argv[1:]

# pass argument to the program if video input or camera input
opts, args = getopt.getopt(argv, "hd:")
dir = None


for opt, arg in opts:
    if opt == '-h':
        print('ProcessFiles.py -d <directory>')
        sys.exit()
    elif opt == '-d':
        dir = arg

if dir is None:
    print('Direktorij ni podan!')
    sys.exit()

input_folder = os.path.join("Data", dir)
output_folder = os.path.join("Output", dir)

if not os.path.exists(output_folder):
    os.mkdir(output_folder)

for video_name in os.listdir(input_folder):
    print(f"Working on {video_name}..")

    input_video_path = os.path.join(input_folder, video_name)
    output_video_path = os.path.join(output_folder, "output_" + video_name)

    DetectOnFile.Detect(input_video_path, output_video_path)

    print("saved to " + output_video_path)
    print(f"Done with {video_name}..\n")