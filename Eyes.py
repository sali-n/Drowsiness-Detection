from scipy.spatial import distance
class Eyes:
    """Eyes class for checking eye status (open, close)."""
    def __init__(self):
        """Initialise."""
        # self.left_eye_indices = [33, 246, 7, 160, 161, 163, 144, 145, 153, 159, 154, 155, 157, 158, 173, 133]
        # self.right_eye_indices = [373, 374, 390, 249, 263, 388, 384, 385, 386, 387, 380, 381, 382, 362, 398, 466]
        # self.right_EAR_indices = [(33, 133), (145, 159), (153, 158), (144, 160)]
        # self.left_EAR_indices = [(263, 362), (380, 385), (374, 386), (373, 387)]
        self.right_EAR_indices = [(33, 133), (160, 144), (158, 153)]
        self.left_EAR_indices = [(362, 263), (385, 380), (387, 373)]

        self.left_EAR = 0
        self.right_EAR = 0
        self.ear = 0

        self.right_maxopen = 0
        self.right_minclosed = 1
        self.left_maxopen = 0
        self.left_minclosed = 1

        self.threshold = 0
        
        self.img_h = 0
        self.img_w = 0

        self.EYE_CLOSED = False
        self.blinks = 0
        self.BLINKS_FRAME = {True: 0, False: 1}
        self.prev_blinks = 0
        self.history = []

    def perclos(self):
        """Return PERCLOS."""
        if self.pupil_area_left != 0:
            self.history.append(self.ear / ((self.pupil_area_left + self.pupil_area_right) / 2))
        if len(self.history) >= 20 and len(self.history) < 50:
            return (sum(self.history)/len(self.history))
        elif len(self.history) > 50:
            self.history = []

        return 0

    def blinking_freq(self, time): 
        """Get no. of blinks per second."""
        freq = 0
        if time % 50 == 0:
            freq = (self.blinks - self.prev_blinks)/50
            self.prev_blink = self.blinks
        return freq

    def eye_closed_duration(self):
        """Checks if eye closed for 20 consecutive frames."""
        if self.BLINKS_FRAME[True] + self.BLINKS_FRAME[False] == 51:  
            self.BLINKS_FRAME = {True: 0, False: 1}
        
        return self.BLINKS_FRAME[True] / (self.BLINKS_FRAME[True] + self.BLINKS_FRAME[False])

    def cirular(self,landmarks):
        """Get Eye Circularity."""
        left_distances = [distance.euclidean((landmarks[i].x*self.img_w, landmarks[i].y*self.img_h), (landmarks[j].x*self.img_w, landmarks[j].y*self.img_h))
                          for i, j in [(246,35),(159,160),(157,173),(173,133),(133,155),(154,153),(144,145),(7,163),(35,7)]]
        
        right_distances = [distance.euclidean((landmarks[i].x*self.img_w, landmarks[i].y*self.img_h), (landmarks[j].x*self.img_w, landmarks[j].y*self.img_h))
                           for i, j in [(368,362),(385,384),(386,387),(381,380),(330,373),(373,374),(362,382)]]

        self.pupil_area_right = ((distance.euclidean((landmarks[386].x*self.img_w, landmarks[386].y*self.img_h), (landmarks[380].x*self.img_w, landmarks[380].y*self.img_h)) /2)**2) * (22/7)
        self.pupil_area_left =  ((distance.euclidean((landmarks[159].x*self.img_w, landmarks[159].y*self.img_h), (landmarks[153].x*self.img_w, landmarks[153].y*self.img_h)) /2)**2) * (22/7)

        ec_right = (4*(22/7)*self.pupil_area_right) / (sum(right_distances)**2)
        ec_left = (4*(22/7)*self.pupil_area_left) / (sum(left_distances)**2)

        return (ec_left+ec_right) / 2

    def pupil_size(self, landmarks):  
        """Get Pupil."""      
        return ((self.pupil_area_left / distance.euclidean((landmarks[33].x*self.img_w, landmarks[33].y*self.img_h), (landmarks[133].x*self.img_w, landmarks[133].y*self.img_h))) + 
                (self.pupil_area_right / distance.euclidean((landmarks[362].x*self.img_w, landmarks[362].y*self.img_h), (landmarks[263].x*self.img_w, landmarks[263].y*self.img_h)))) / 2            

    def eyebrow(self, landmarks):
        left_distances = [distance.euclidean((landmarks[i].x*self.img_w, landmarks[i].y*self.img_h), (landmarks[j].x*self.img_w, landmarks[j].y*self.img_h))
                          for i, j in [(221,243),(180,243)]]
        
        right_distances = [distance.euclidean((landmarks[i].x*self.img_w, landmarks[i].y*self.img_h), (landmarks[j].x*self.img_w, landmarks[j].y*self.img_h))
                           for i, j in [(442,463),(441,463)]]
        
        return ((sum(left_distances) / 2) + (sum(right_distances) / 2)) / 2

    def __update_aspect_ratios(self, landmarks):
        """Update aspect ratio of both eyes."""

        left_distances = [distance.euclidean((landmarks[i].x*self.img_w, landmarks[i].y*self.img_h), (landmarks[j].x*self.img_w, landmarks[j].y*self.img_h))
                          for i, j in self.left_EAR_indices]
        right_distances = [distance.euclidean((landmarks[i].x*self.img_w, landmarks[i].y*self.img_h), (landmarks[j].x*self.img_w, landmarks[j].y*self.img_h))
                           for i, j in self.right_EAR_indices]
        
        self.left_EAR = sum(left_distances[1:]) / (2 * left_distances[0])
        self.right_EAR = sum(right_distances[1:]) / (2 * right_distances[0])

        self.ear = (self.left_EAR + self.right_EAR) / 2

    def check_blink(self,landmarks):
        """Check if subject is blinking."""
        current_status = self.get_status(landmarks)

        if self.EYE_CLOSED and current_status:
            self.blinks += 1
            self.EYE_CLOSED = False
        elif not self.EYE_CLOSED and not current_status:
            self.EYE_CLOSED = True
        
        self.BLINKS_FRAME[not current_status] += 1
        
        return not current_status

    def get_status(self, landmarks):
        """Return status of the eyes."""
        self.__update_aspect_ratios(landmarks)
        if self.ear < self.threshold*0.9:
            return False
        return True
    
    def update_threshold(self,landmarks,k):
        """Threshold of Right and Left EAR updated"""
        self.__update_aspect_ratios(landmarks)
        
        # Reset Values to take into account different facial orientation.
        if k%60 == 0: #k%10 <= 0.1
            self.right_maxopen = 0
            self.left_maxopen = 0
            self.left_minclosed = 1
            self.right_minclosed = 1

        self.right_maxopen = max(self.right_maxopen,self.right_EAR)
        self.left_maxopen = max(self.left_maxopen,self.left_EAR)
        self.left_minclosed = min(self.left_minclosed,self.left_EAR)
        self.right_minclosed = min(self.right_minclosed,self.right_EAR)

        # Threshold Calculation.
        self.threshold = (((self.right_minclosed + self.right_maxopen) / 2) + ((self.left_minclosed + self.left_maxopen) / 2)) / 2 
        