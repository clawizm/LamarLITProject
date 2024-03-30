            #encapsulate into hand detection function, CALLED ON EACH OBJECT IN FRAME
            try:
                cropped_image = self.frame[self.ymin: self.ymax, self.xmin: self.xmax]
                cropped_image_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
                results = self.hands.process(cropped_image_rgb)
                if results.multi_hand_landmarks:
                    hands_in_frame = True
                    for hand_landmarks in results.multi_hand_landmarks:

                        mp.solutions.drawing_utils.draw_landmarks(cropped_image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                        landmark_list = calc_landmark_list(cropped_image, hand_landmarks)

                        # Conversion to relative coordinates / normalized coordinates
                        pre_processed_landmark_list = pre_process_landmark(
                            landmark_list)

                        self.hand_sign_id = self.keypoint_classifier(pre_processed_landmark_list)
                    if self.previous_gestures != self.keypoint_classifier_labels[self.hand_sign_id]:   
                        if self.previous_gestures and self.gesture_start_time:
                            duration = time.time() - self.gesture_start_time 
                            print(f'Detected {self.previous_gestures} for {duration}')
                            if duration > 3 and self.previous_gestures == 'Love':
                                self.gui_window.write_event_value(f"-CAMERA_{self.camera_index}_TURNONALLLEDs-", 'Update')
                            self.gesture_start_time = None
                        self.gesture_start_time = time.time()
                        self.previous_gestures = self.keypoint_classifier_labels[self.hand_sign_id]
            except:
                print('Hand Error')
        ###ENCAPSULATE INTO FUNCTION, USED TO HANDLE IF THERE IS NO ONE IN FRAME OR NO HANDS IN FRAME
        if len(classes) == 0 or not hands_in_frame:
            if self.previous_gestures and self.gesture_start_time:
                duration = time.time() - self.gesture_start_time 
                print(f'Detected {self.previous_gestures} for {duration}')
                if duration > 3 and self.previous_gestures == 'Love':
                    self.gui_window.write_event_value(f"-CAMERA_{self.camera_index}_TURNONALLLEDs-", True)
                self.gesture_start_time = None
            self.previous_gestures = None