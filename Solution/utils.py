import cv2
import os
import shutil


def capture_frames(video_filename: str,
                   output_directory: str = "frames") -> None:
    """Capture frames from a video file and save them as images in the output directory."""
    # check output directory exists or not
    output_directory = os.path.join(output_directory, video_filename)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory, exist_ok=True)

    alternate_path = video_filename
    if not os.path.exists(video_filename):
        alternate_path = os.path.join("..", "Samples", video_filename)
        if not os.path.exists(alternate_path):
            print(f"File '{video_filename}' not found.")
            return

    # Open the video file
    video = cv2.VideoCapture(alternate_path)
    print(f"Start capture frames in {alternate_path}")

    frame_counter = 0
    while True:
        ret, frame = video.read()
        if ret:
            filename = os.path.join(
                output_directory,
                f"{video_filename.split('.')[0]}_{frame_counter}.jpg")
            cv2.imwrite(filename, frame)
            frame_counter += 1
        else:
            break

    video.release()
    cv2.destroyAllWindows()
    print(f"Total number of frames in {video_filename}: {frame_counter}")


def create_dataset_from_structure(
        filename: str = "dataset_structure.txt") -> None:
    """Create dataset from the structure of the dataset."""
    output_directory = "dataset"

    with open(filename, 'r') as file:
        lines = file.readlines()

    for line in lines:
        parts = line.strip().split('\\')
        filename = parts[-1]
        subdirs = parts[1:-1]

        target_dir = os.path.join(output_directory, *subdirs)
        os.makedirs(target_dir, exist_ok=True)

        source_image_path = os.path.join('frames',
                                         filename.split('_')[0] + ".mp4",
                                         filename)

        target_image_path = os.path.join(target_dir, filename)

        if os.path.exists(source_image_path):
            shutil.copyfile(source_image_path, target_image_path)
        else:
            print(
                f"Warning: {source_image_path} does not exist and will be skipped."
            )

    file.close()


def get_filenames_in_folder(folder: str = 'dataset') -> list:
    """Get all filenames in a dataset."""
    file_list = list()
    for root, _, files in os.walk(folder):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list


def get_frame_list(video_filename: str) -> list:
    """Get the list of frames from a video file."""
    frame_list = list()
    root_folder = os.path.join("frames", video_filename)

    if not os.path.exists(root_folder):
        return None

    for frame in os.listdir(root_folder):
        frame_list.append(os.path.join(root_folder, frame))

    return sorted(frame_list)


def write_dataset_log(filelist: list,
                      filename: str = "dataset_structure.txt") -> None:
    """Save the input file list to a text file."""
    with open(filename, "w") as file:
        for item in filelist:
            file.write("%s\n" % item)
    file.close()


if __name__ == "__main__":
    videos = ["01.mp4", "02.mp4", "03.mp4"]
    for video in videos:
        capture_frames(video)
    create_dataset_from_structure()
