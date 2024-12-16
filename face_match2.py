import cv2
import face_recognition
import time
from scipy.spatial.distance import euclidean

# Function to preprocess image and detect faces
def preprocess_and_detect_faces(image):
    # Convert image to RGB (face_recognition library expects RGB images)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Detect faces in the image
    face_locations = face_recognition.face_locations(rgb_image)
    
    return face_locations

# Function to extract facial embeddings
def extract_embeddings(image, face_locations):
    # Convert image to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Extract facial embeddings
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
    
    return face_encodings

# Function to calculate similarity between embeddings
def calculate_similarity(embedding1, embedding2):
    # Calculate Euclidean distance between embeddings
    distance = face_recognition.face_distance([embedding1], embedding2)
    
    # Convert distance to percentage match
    percentage_match = (1 - distance[0]) * 100
    
    return percentage_match

def main():
    # Load the video capture object for camera feed
    video_capture = cv2.VideoCapture(0)

    # Load the document image for comparison
    document_image_path = r"D:\Divakar docoments\Divakar pan card.jpg"
    document_image = cv2.imread(document_image_path)

    # Preprocess document image and extract faces
    document_face_locations = preprocess_and_detect_faces(document_image)
    document_embeddings = extract_embeddings(document_image, document_face_locations)

    # Variables to track total match percentage and number of frames processed
    total_match_percentage = 0
    num_frames = 0
    start_time = time.time()
    print_counter = 0

    # Process each frame from the camera feed
    while True:
        # Capture frame-by-frame from camera
        ret, frame = video_capture.read()
        if not ret:
            break
        
        # Preprocess video frame and detect faces
        face_locations = preprocess_and_detect_faces(frame)
        
        # Extract embeddings from video frame
        frame_embeddings = extract_embeddings(frame, face_locations)
        
        # Initialize variables to store match status and percentage for the current frame
        frame_match_status = "Not Match"
        frame_match_percentage = 0
        
        # Compare embeddings of document face with faces in the video frame
        for i, document_embedding in enumerate(document_embeddings):
            for frame_embedding, (top, right, bottom, left) in zip(frame_embeddings, face_locations):
                # Calculate similarity between embeddings
                match_percentage = calculate_similarity(document_embedding, frame_embedding)
                
                # Update match status and percentage if a better match is found
                if match_percentage > frame_match_percentage:
                    frame_match_percentage = match_percentage

                    if match_percentage > 50:
                        frame_match_status = "Match"
                    else:
                        frame_match_status = "Not Match"
                    
                    # Draw rectangle around the face
                    cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)

                    # Check for liveness by detecting facial movements
                    landmarks = face_recognition.face_landmarks(frame, [face_locations[i]])[0]
                    eye_aspect_ratio = (euclidean(landmarks["left_eye"][1], landmarks["left_eye"][5]) +
                                        euclidean(landmarks["left_eye"][2], landmarks["left_eye"][4])) / (
                                               2 * euclidean(landmarks["left_eye"][0], landmarks["left_eye"][3]))
                    mouth_aspect_ratio = (euclidean(landmarks["top_lip"][2], landmarks["bottom_lip"][10]) +
                                          euclidean(landmarks["top_lip"][4], landmarks["bottom_lip"][8])) / (
                                                 2 * euclidean(landmarks["top_lip"][0], landmarks["top_lip"][6]))

                    # Define thresholds for eye and mouth aspect ratios
                    EAR_THRESHOLD = 0.2
                    MAR_THRESHOLD = 0.5

                    # Check for eye blink and mouth open
                    if eye_aspect_ratio < EAR_THRESHOLD or mouth_aspect_ratio > MAR_THRESHOLD:
                        cv2.putText(frame, "Live", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    else:
                        cv2.putText(frame, "Not Live", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Update total match percentage and number of frames processed
        total_match_percentage += frame_match_percentage
        num_frames += 1

        # Display match status and percentage on the frame
        cv2.putText(frame, f"Match Percentage: {frame_match_percentage:.2f}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, f"Match Status: {frame_match_status}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Video', frame)
        
        # Print match status and percentage on the output side
        print(f"Match Percentage: {frame_match_percentage:.2f}%")
        print(f"Match Status: {frame_match_status}")
        
        # Break the loop when 'q' is pressed
        if cv2.waitKey(20) & 0xFF == ord('q'):  # Decreased waitKey to 20ms for smoother experience
            break

        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        if elapsed_time >= 10 or elapsed_time >= 15:
            # Calculate total average matching
            if num_frames > 0:
                total_average_matching = total_match_percentage / num_frames
                print("----------------------------------------------------------------------------")
                print(f"\n Number of time check : {num_frames}")
                print(f"\n Total Average Matching: {total_average_matching:.2f}%")  
                print(f" Match Status: {frame_match_status}")  
                print("----------------------------------------------------------------------------")

                print_counter += 1
                if print_counter >= 2 or print_counter >= 3:  # Check if printed 2 or 3 times
                    # Release the video capture object
                    video_capture.release()
                    cv2.destroyAllWindows()
                    break

            # Reset variables for next calculation
            start_time = time.time()
            total_match_percentage = 0
            num_frames = 0

    # Release the video capture object
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
