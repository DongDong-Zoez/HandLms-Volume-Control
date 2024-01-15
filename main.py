import argparse
import numpy as np
import time
import cv2 
import mediapipe as mp

from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

def argparser() -> argparse.Namespace:
    """
    Function that defines and parses command line arguments.
    """
    parser = argparse.ArgumentParser(description="Face mask detection using YOLOv8 and OpenCV")
    parser.add_argument("-dc", "--detection_confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
    parser.add_argument("-tc", "--tracking_confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
    parser.add_argument("--vol_diff", type=float, default=2., help="minimum vol for confiming the volume")
    parser.add_argument("--check_time", type=float, default=2., help="maximum time for confirming the volume")
    parser.add_argument("--num_hands", type=int, default=1, help="number hands to detect")
    parser.add_argument("--h", type=int, default=720, help="window height for webcamera")
    parser.add_argument("--w", type=int, default=1280, help="window width for webcamera")
    return parser.parse_args()

class AIVideo(object):

    def __init__(self, h, w, min_detection_confidence, min_tracking_confidence, num_hands, vol_diff, check_time):

        # Video frame dimensions
        self.h = h
        self.w = w

        # MediaPipe Hands module
        self.mpHands = mp.solutions.hands

        # Confidence thresholds and number of hands to detect
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.num_hands = num_hands

        # MediaPipe Drawing utilities
        self.mpDraw = mp.solutions.drawing_utils
        self.mpDrawStyle = mp.solutions.drawing_styles

        # Get audio volume controls
        self.volume, self.vmin, self.vmax = self.getVolume

        # Parameters for volume control
        self.vol_diff = vol_diff
        self.check_time = check_time

    @property
    def getVolume(self):
        # Get audio volume controls using pycaw
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = cast(interface, POINTER(IAudioEndpointVolume))

        # Get volume range
        vmin, vmax, vinc = volume.GetVolumeRange()
        return volume, vmin, vmax

    def liveStream(self):
        # Open webcam for live streaming
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.h)

        # Rectangle dimensions for volume indicator
        rctH = int(self.h * 0.8)
        rctW = 50

        # Initialize time variables
        cTime = 0
        pTime = 0
        Time = 0

        # Constants for volume control
        pVol = -100
        bot = 0.04035360699999999
        c = 32.278481179064414

        # Initialize MediaPipe Hands module
        with self.mpHands.Hands(
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
            max_num_hands=self.num_hands,
        ) as hands:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    print("Empty frame.")
                    continue

                # Process frame with MediaPipe Hands
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.flip(frame, 1)
                result = hands.process(frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                if result.multi_hand_landmarks:
                    for handLms in result.multi_hand_landmarks:
                        # Draw hand landmarks and connections
                        self.mpDraw.draw_landmarks(
                            frame,
                            handLms,
                            self.mpHands.HAND_CONNECTIONS,
                            self.mpDrawStyle.get_default_hand_landmarks_style(),
                            self.mpDrawStyle.get_default_hand_connections_style(),
                        )

                        # Extract hand landmarks for volume control
                        landmarks = [(lm.x * self.w, lm.y * self.h) for lm in handLms.landmark]
                        dist = ((landmarks[4][0] - landmarks[8][0]) ** 2 + (landmarks[4][1] - landmarks[8][1]) ** 2) ** .5
                        middle = ((landmarks[4][0] + landmarks[8][0]) / 2, (landmarks[4][1] + landmarks[8][1]) / 2)
                        # Set a landmark for camera deep ratio control
                        mark = ((landmarks[0][0] - landmarks[1][0]) ** 2 + (landmarks[0][1] - landmarks[1][1]) ** 2) ** .5

                        # Visualize key points
                        cv2.circle(frame, (int(landmarks[4][0]), int(landmarks[4][1])), 10, (255, 0, 255), -1)
                        cv2.circle(frame, (int(landmarks[8][0]), int(landmarks[8][1])), 10, (255, 0, 255), -1)
                        cv2.circle(frame, (int(middle[0]), int(middle[1])), 10, (255, 0, 255), -1)
                        cv2.line(frame, (int(landmarks[4][0]), int(landmarks[4][1])),
                                 (int(landmarks[8][0]), int(landmarks[8][1])), (255, 0, 255), 5)

                        # Calculate volume based on hand movements
                        ratio = dist / mark
                        low, high = 0.3, 2.5
                        vol = np.clip((ratio - low) / (high - low), 0, 1) * 100

                    # Display volume indicator
                    cv2.rectangle(frame, (rctW, rctH), (rctW + 25, rctH + 100), (255, 0, 0), 3)
                    cv2.rectangle(frame, (rctW, rctH + 100 - int(vol)), (rctW + 25, rctH + 100), (255, 0, 0), -1)
                    cv2.putText(frame, f"vol: {int(vol)}", (rctW - 20, rctH - 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (255, 0, 0), 3)

                    # Check for significant volume change and terminate
                    volDiff = vol - pVol
                    if volDiff < self.vol_diff:
                        Time += time.time() - pTime
                        if Time > self.check_time:
                            return
                    else:
                        Time = 0
                    pVol = vol

                    # Set the system volume based on hand movement
                    self.volume.SetMasterVolumeLevel(np.log10(max(vol / 100, bot)) * c, None)

                # Calculate and display frames per second
                cTime = time.time()
                fps = 1 / (cTime - pTime)
                pTime = cTime
                cv2.putText(frame, f"FPS : {int(fps)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

                # Display the frame
                cv2.imshow('img', frame)

                # Check for keyboard interrupt to stop the program
                if cv2.waitKey(1) == ord('q'):
                    print("Keyboard Interrupt.")
                    return
            cap.release()

def main():
    args = argparser()
    live = AIVideo(
        h=args.h,
        w=args.w,
        min_detection_confidence=args.detection_confidence,
        min_tracking_confidence=args.tracking_confidence,
        num_hands=args.num_hands,
        vol_diff=args.vol_diff,
        check_time=args.check_time,
    )
    live.liveStream()

if __name__ == "__main__":
    main()