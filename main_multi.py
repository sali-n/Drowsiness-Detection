import cv2
import mediapipe as mp

# Feature Imports.
from time import perf_counter
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance
from Eyes import Eyes

# Model Imports.
import pickle
from xgboost import XGBClassifier

# Training mports.
import os
import concurrent.futures

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

def moe(mar, ear):
    """Find mouth over eye ratio."""
    return ear and mar / ear or 0

def mouth_ratio(landmarks, h, w):
    """Find mouth ratio."""
    mouth_distances = [distance.euclidean((landmarks[i].x*w, landmarks[i].y*h), (landmarks[j].x*w, landmarks[j].y*h))
                    for i, j in [(61,291),(0,17),(39,181),(269,405)]]
    
    return (sum(mouth_distances[1:]) / (3*mouth_distances[0]))

def draw_mesh(frame, landmarks):
    """Draw mesh on the face."""
    mp_drawing.draw_landmarks(
        image=frame,
        landmark_list=landmarks,
        connections=mp_face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(
            color=(0, 255, 0),
            thickness=1,
            circle_radius=1),
        connection_drawing_spec=mp_drawing.DrawingSpec(
            color=(0, 255, 0),
            thickness=1)
    )

def get_video(file):
    """Main Code to detect Drowsiness."""
    EYES = Eyes()
    scaler = StandardScaler() 
    drowsyhist = []
    drowsy = False
    model = pickle.load(open(r'Drowsiness-Detection/final.pkl', "rb"))     

    cap = cv2.VideoCapture(file) 

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.8) as face_mesh:
        n_frames = 0
        while True:
            ret, frame = cap.read()
            while ret:
                start_fps=perf_counter()
                
                # Initialize Values.
                MR = 0
                moer = 0
                perclo = 0
                eye_circ = 0
                pupil = 0
                eyebrow = 0

                # Brighten and Redo Contrast of Frame.
                cv2.normalize(frame, frame, 0, 255, cv2.NORM_MINMAX)
                
                # Process Frame.
                frame.flags.writeable = False
                results = face_mesh.process(frame)
                frame.flags.writeable = True
                
                h,w,_= frame.shape
                EYES.img_h = h
                EYES.img_w = w

                # Get Features.
                if results.multi_face_landmarks:
                    face_landmarks = results.multi_face_landmarks[0]
                    EYES.update_threshold(face_landmarks.landmark, n_frames)
                    EYES.check_blink(face_landmarks.landmark)
                    eye_circ = EYES.cirular(face_landmarks.landmark)
                    perclo  = EYES.perclos()
                    pupil = EYES.pupil_size(face_landmarks.landmark)
                    eyebrow = EYES.eyebrow(face_landmarks.landmark)
                    MR = mouth_ratio(face_landmarks.landmark, h, w)
                    moer = moe(MR, EYES.ear)
                freq = EYES.blinking_freq(n_frames)
                eye_closed = EYES.eye_closed_duration()

                # Scale Features. 
                if n_frames < 200: # Time to calibrate.
                    # scaler.partial_fit([[MR, eye_closed, freq, perclo, eye_circ, pupil, eyebrow, moer, EYES.ear]])
                    scaler.partial_fit([[eye_closed, perclo, pupil, moer]])
                else:
                    # feat = scaler.transform([[MR, eye_closed, freq, perclo, eye_circ, pupil, eyebrow, moer, EYES.ear]]) #moe, ear
                    feat = scaler.transform([[eye_closed, perclo, pupil, moer]])
                    drowsy = model.predict(feat)
                    drowsyhist.append(drowsy)
                    if len(drowsyhist) == 80:
                        drowsyhist = []
                if drowsyhist.count(True) > 0.8*len(drowsyhist):  # Edit this value to get more responsive.
                    drowsy = 'Drowsy'
                else:
                    drowsy = 'Not'
                fps = 1/(perf_counter()-start_fps)
                
                cv2.putText(frame, f"No.of Blinks: {EYES.blinks}", (410, 50), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(frame,f'Drowsy: {drowsy}',(410,20), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(frame,f'FPS: {int(fps)}',(20,450), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2)
                # cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
                frame = cv2.resize(frame,(960,540))  # 960, 540 - 300, 300
                
                # cv2.imshow(f'{file[49:55]}_{file[-6:-4]}', frame)
                cv2.imshow('hey', frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

                n_frames += 1
                ret, frame = cap.read()

            cap.release()
            cv2.destroyAllWindows()
            break

def parallel(file):
    """Run Multiple Videos in Parallel."""
    with concurrent.futures.ProcessPoolExecutor(max_workers = 4) as exe:
        exe.map(get_video, file)


if __name__ == '__main__':
    # folder = r'D:\Final Year Project Dataset\RLDD Dataset\Fold4'
    # files = [os.path.join(folder, filename) for filename in os.listdir(folder)]
    # parallel(files)
    get_video(0)