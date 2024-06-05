import cv2
import numpy as np

# Preparation function to detect edges
def preparation(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (25, 25), 0)
    _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(thresh, 70, 180)
    return edges

# Function to find the most symmetric region in the first frame
def find_most_symmetric_region(image, window_size):
    h, w = image.shape
    x_center = w // 2
    max_score = -1
    best_region = None
    for y in range(0, h - window_size + 1, window_size // 2):
        for x in range(window_size // 2, w // 2, window_size // 2):
            w_half = min(window_size // 2, x_center, w - x_center)
            left_region = image[y:y+window_size, x_center-w_half:x_center]
            right_region = image[y:y+window_size, x_center:x_center+w_half]
            flipped_right_region = cv2.flip(right_region, 1)
            score = np.sum(left_region == flipped_right_region)
            if score > max_score:
                max_score = score
                best_region = (x_center - w_half, y, x_center + w_half, y + window_size)
    return best_region

# Function to draw the symmetric region
def draw_symmetric_region(image, region):
    x1, y1, x2, y2 = region
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return image

# Function to calculate optical flow and estimate depth change
def calculate_optical_flow(prev_frame, next_frame, region):
    x1, y1, x2, y2 = region
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

    prev_region = prev_gray[y1:y2, x1:x2]
    next_region = next_gray[y1:y2, x1:x2]

    flow = cv2.calcOpticalFlowFarneback(prev_region, next_region, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    return np.mean(magnitude)

# Main function to process the video and detect depth changes
def process_video(video_path, window_size):
    cap = cv2.VideoCapture(video_path)

    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not read the video file.")
        return

    edges = preparation(first_frame)
    best_region = find_most_symmetric_region(edges, window_size)
    x1, y1, x2, y2 = best_region

    output_img = draw_symmetric_region(first_frame.copy(), best_region)
    cv2.imshow('Most Symmetric Region in First Frame', output_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    prev_frame = first_frame
    depth_changes = []

    frame_number = 0
    while cap.isOpened():
        ret, next_frame = cap.read()
        if not ret:
            break

        depth_change = calculate_optical_flow(prev_frame, next_frame, best_region)
        depth_changes.append((depth_change, frame_number))

        prev_frame = next_frame
        frame_number += 1

    cap.release()

    # Sort depth changes in descending order
    depth_changes.sort(key=lambda x: -x[0])

    # Select frames ensuring they are not within 20 frames of each other
    selected_frames = []
    i = 0
    while len(selected_frames) < 2 and i < len(depth_changes):
        frame_number = depth_changes[i][1]
        if not selected_frames or abs(frame_number - selected_frames[-1]) > 30:
            selected_frames.append(frame_number)
        i += 1

    # Ensure the first frame is smaller than the second frame
    if len(selected_frames) == 2 and selected_frames[0] > selected_frames[1]:
        selected_frames.reverse()

    print("Selected frames:", selected_frames)
    return selected_frames

# Example usage
if __name__ == "__main__":
    video_path = '00.mp4'  # Path to your video file
    window_size = 650  # Adjust the window size as needed
    selected_frames = process_video(video_path, window_size)
    print(f"Selected frames: {selected_frames}")
