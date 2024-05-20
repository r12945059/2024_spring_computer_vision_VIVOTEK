import cv2
import os


def capture_frames(video_filename: str,
                   output_directory: str = "frames") -> None:
    """Capture frames from a video file and save them as images in the output directory."""
    # check output directory exists or not
    output_directory = os.path.join(output_directory, video_filename)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory, exist_ok=True)

    # Open the video file
    video = cv2.VideoCapture(video_filename)
    print(f"Start capture frames in {video_filename}")

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


def get_frame_list(video_filename: str) -> list:
    """Get the list of frames from a video file."""
    frame_list = list()
    root_folder = os.path.join("frames", video_filename)

    if not os.path.exists(root_folder):
        return None

    for frame in os.listdir(root_folder):
        frame_list.append(os.path.join(root_folder, frame))

    return sorted(frame_list)


if __name__ == "__main__":
    print("hello world==")
    frame_list = get_frame_list("01.mp4")
    print(*frame_list, sep="\n")
