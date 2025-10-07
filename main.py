from ultralytics import YOLO
import cv2
import os
import tempfile
import time
import numpy as np
from collections import defaultdict, deque
import re
import requests
import threading
import queue
from datetime import datetime

import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, 'models')

CONFIG = {
    'PLATE_MODEL_PATH': os.path.join(MODELS_DIR, 'plate_detection.pt'),
    'CHAR_MODEL_PATH': os.path.join(MODELS_DIR, 'character_recognition.pt'),
    'VIDEO_SOURCE': "rtsp://admin:Admin%4001%21@192.168.100.132:554/Streaming/Channels/101",
    'CONFIDENCE_THRESHOLD': 0.5,
    'TRACKER_MAX_AGE': 30,
    'TRACKER_MIN_HITS': 3,
    'TRACKER_IOU_THRESHOLD': 0.3,
    'FRAME_SKIP': 3,
    'SAVE_DETECTIONS': True,
    'SHOW_INDIVIDUAL_PLATES': True,
    'DATABASE_ENABLED': True,
    'API_BASE_URL': "http://172.30.30.96:2500/RFID",
    'DEVICE': '01',
    'IN_OUT': 'I',
    'DB_ASYNC_MODE': True,
    'MIN_CONFIDENCE_FOR_DB': 0.75,
    'MIN_HITS_FOR_DB': 40,
}

class DatabaseManager:
    def __init__(self, base_url, cam_code, device, in_out):
        self.base_url = base_url
        self.cam_code = cam_code
        self.device = device
        self.in_out = in_out
        self.insert_endpoint = f"{base_url}/PostRFID"
        self.select_endpoint = f"{base_url}/GetCamDetails"
        
        self.insert_queue = queue.Queue()
        self.is_running = True
        self.insert_thread = threading.Thread(target=self._process_insert_queue, daemon=True)
        self.insert_thread.start()
        
        self.stats = {'total_inserts': 0, 'successful_inserts': 0, 'failed_inserts': 0}
        print(f"🗄️  Database Manager initialized - Camera {device} ({in_out})")
    
    def insert_plate_detection(self, plate_number, confidence, track_id, sync=False):
        detection_data = {'plate_number': plate_number}
        
        if sync:
            return self._perform_insert(detection_data)
        else:
            self.insert_queue.put(detection_data)
            return True
    
    def _perform_insert(self, detection_data):
        try:
            params = {
                'CamCode': detection_data['plate_number'],
                'Device': self.device,
                'InOut': self.in_out
            }
            
            response = requests.get(self.insert_endpoint, params=params, timeout=10)
            self.stats['total_inserts'] += 1
            
            if response.status_code in [200, 201]:
                self.stats['successful_inserts'] += 1
                print(f"✅ Plate '{detection_data['plate_number']}' saved to database")
                return True
            else:
                self.stats['failed_inserts'] += 1
                return False
        except Exception as e:
            self.stats['failed_inserts'] += 1
            print(f"❌ Database error: {e}")
            return False
    
    def _process_insert_queue(self):
        while self.is_running:
            try:
                detection_data = self.insert_queue.get(timeout=1)
                self._perform_insert(detection_data)
                self.insert_queue.task_done()
            except queue.Empty:
                continue
    
    def get_stats(self):
        return self.stats.copy()
    
    def shutdown(self):
        self.is_running = False
        if self.insert_thread.is_alive():
            self.insert_thread.join(timeout=2)

class PlateTracker:
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = {}
        self.next_id = 1
        self.frame_count = 0
        self.saved_to_db = set()
        
    def update(self, detections):
        self.frame_count += 1
        updated_tracks = {}
        
        for track_id, track in self.tracks.items():
            best_match_idx, best_iou = -1, 0
            for i, det in enumerate(detections):
                iou = self.calculate_iou(track['bbox'], det['bbox'])
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_match_idx = i
            
            if best_match_idx >= 0:
                det = detections[best_match_idx]
                track['bbox'] = det['bbox']
                track['text_history'].append(det['text'])
                track['confidence_history'].append(det['confidence'])
                track['last_seen'] = self.frame_count
                track['hits'] += 1
                track['crop'] = det['crop']
                track['consensus_text'] = self.get_consensus_text(track['text_history'])
                track['avg_confidence'] = np.mean(track['confidence_history'])
                updated_tracks[track_id] = track
                detections.pop(best_match_idx)
            elif self.frame_count - track['last_seen'] < self.max_age:
                updated_tracks[track_id] = track
        
        for det in detections:
            new_track = {
                'bbox': det['bbox'],
                'text_history': [det['text']],
                'confidence_history': [det['confidence']],
                'consensus_text': det['text'],
                'avg_confidence': det['confidence'],
                'hits': 1,
                'last_seen': self.frame_count,
                'first_seen': self.frame_count,
                'crop': det['crop'],
                'saved_to_db': False
            }
            updated_tracks[self.next_id] = new_track
            self.next_id += 1
        
        self.tracks = updated_tracks
        return self.tracks
    
    def calculate_iou(self, box1, box2):
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0
    
    def get_consensus_text(self, text_history):
        if not text_history:
            return ""
        
        recent_texts = text_history[-10:]
        text_counts = {}
        for text in recent_texts:
            if text and text != "No text detected":
                text_counts[text] = text_counts.get(text, 0) + 1
        
        if not text_counts:
            return recent_texts[-1] if recent_texts else ""
        
        consensus = max(text_counts.items(), key=lambda x: x[1])[0]
        
        formatted_plates = []
        for text, count in text_counts.items():
            if re.match(r'^[A-Z]{2,3}-\d{1,4}[A-Z]?$', text):
                formatted_plates.append((text, count))
        
        if formatted_plates:
            return max(formatted_plates, key=lambda x: x[1])[0]
        
        return consensus

    def should_save_to_database(self, track):
        return not track.get('saved_to_db', False)
    
    def calculate_plate_similarity(self, plate1, plate2):
        if not plate1 or not plate2:
            return 0.0
        
        if plate1 == plate2:
            return 1.0
        
        confusions = {
            'K': 'X', 'X': 'K', 'D': 'O', 'O': 'D', 
            'B': '8', '8': 'B', 'P': 'B', 'G': 'C', 'C': 'G'
        }
        
        variations = [plate1]
        for i, char in enumerate(plate1):
            if char in confusions:
                variation = plate1[:i] + confusions[char] + plate1[i+1:]
                variations.append(variation)
        
        if plate2 in variations:
            return 0.9
        
        if len(plate1) == len(plate2):
            matches = sum(1 for a, b in zip(plate1, plate2) if a == b)
            return matches / len(plate1)
        
        return 0.0

    def should_save_to_database(self, track_id, track):
        plate_key = f"{track_id}_{track['consensus_text']}"
        if plate_key in self.saved_to_db:
            return False
        
        if track['hits'] < self.min_hits:
            return False
        
        if not track.get('is_valid', False):
            return False
        
        if track['avg_confidence'] < CONFIG.get('MIN_CONFIDENCE_FOR_DB', 0.7):
            return False
        
        self.saved_to_db.add(plate_key)
        return True

class SriLankanPlateValidator:
    def __init__(self):
        self.class_to_char = {
            0: '-', 1: '0', 2: '1', 3: '2', 4: '3', 5: '4', 6: '5', 7: '6', 8: '7', 9: '8', 10: '9',
            11: 'A', 12: 'B', 13: 'C', 14: 'D', 15: 'E', 16: 'F', 17: 'G', 18: 'H', 19: 'I', 20: 'J',
            21: 'K', 22: 'L', 23: 'M', 24: 'N', 25: 'O', 26: 'P', 27: 'Q', 28: 'R', 29: 'S', 30: 'T',
            31: 'U', 32: 'V', 33: 'W', 34: 'X', 35: 'Y', 36: 'Z'
        }
        
        self.patterns = [
            r'^[A-Z]{2}-\d{4}$',
            r'^[A-Z]{3}-\d{4}$'
        ]
        
        self.invalid_patterns = [
            r'^[A-Z]{1}-.*',
            r'^.*-\d{1,3}$',
            r'^.*-\d{5,}$',
            r'^.*-\d{1,4}[A-Z]+.*',
            r'^[A-Z]{4,}-.*',
            r'^.*-[A-Z]{2,}$',
        ]
        
        self.valid_prefixes = []
        letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        for first in letters:
            for second in letters:
                self.valid_prefixes.append(first + second)
        
        for first in letters:
            for second in letters:
                for third in letters:
                    prefix = first + second + third
                    self.valid_prefixes.append(prefix)
        
        self.letter_corrections = {
            '0': 'O', '1': 'I', '2': 'Z', '3': 'E', '4': 'A', '5': 'S', 
            '6': 'G', '7': 'T', '8': 'B', '9': 'P'
        }
        
        self.number_corrections = {
            'O': '0', 'I': '1', 'Z': '2', 'E': '3', 'A': '4', 'S': '5', 
            'G': '6', 'T': '7', 'B': '8', 'P': '9', 'D': '0', 'Q': '0'
        }
        
        self.letter_confusions = {
            'K': ['X', 'H'], 'X': ['K'], 'H': ['K'], 
            'B': ['8', 'P'], 'P': ['B'], '8': ['B'],
            'D': ['O', '0'], 'O': ['D', '0'], '0': ['O', 'D'],
            'C': ['G'], 'G': ['C'], 'F': ['E'], 'E': ['F']
        }
    
    def validate_and_correct(self, plate_text, confidence):
        if not plate_text or plate_text == "No text detected":
            return plate_text, confidence, False
        
        cleaned_text = plate_text.strip().upper()
        corrected_text = self.apply_positional_corrections(cleaned_text)
        formatted_text = self.format_sri_lankan_plate(corrected_text)
        
        is_valid = any(re.match(pattern, formatted_text) for pattern in self.patterns)
        is_invalid = any(re.match(pattern, formatted_text) for pattern in self.invalid_patterns)
        
        prefix_realistic = False
        dash_pos = formatted_text.find('-')
        if dash_pos > 0:
            prefix = formatted_text[:dash_pos]
            prefix_realistic = prefix in self.valid_prefixes
        
        is_valid = is_valid and not is_invalid and prefix_realistic
        adjusted_confidence = confidence * (1.2 if is_valid else 0.8)
        
        return formatted_text, min(adjusted_confidence, 1.0), is_valid
    
    def apply_positional_corrections(self, text):
        if not text:
            return text
        
        if '-' in text:
            parts = text.split('-')
            if len(parts) == 2:
                prefix, suffix = parts
                
                if len(prefix) >= 3:
                    last_char = prefix[-1]
                    if last_char.isalpha() and last_char in self.number_corrections:
                        corrected_char = self.number_corrections[last_char]
                        prefix = prefix[:-1]
                        suffix = corrected_char + suffix
                
                corrected_prefix = ""
                for char in prefix:
                    if char.isdigit():
                        corrected_prefix += self.letter_corrections.get(char, char)
                    else:
                        corrected_prefix += char
                
                corrected_suffix = ""
                for i, char in enumerate(suffix):
                    if char.isalpha():
                        corrected_suffix += self.number_corrections.get(char, char)
                    else:
                        corrected_suffix += char
                
                return f"{corrected_prefix}-{corrected_suffix}"
        
        clean_text = text.replace(' ', '').upper()
        corrected = []
        
        for i, char in enumerate(clean_text):
            if i < 3:
                if char.isdigit():
                    corrected.append(self.letter_corrections.get(char, char))
                else:
                    corrected.append(char)
            else:
                if char.isalpha() and i < len(clean_text) - 1:
                    corrected.append(self.number_corrections.get(char, char))
                else:
                    corrected.append(char)
        
        return ''.join(corrected)

    def format_sri_lankan_plate(self, text):
        if not text:
            return text
        
        clean_text = text.replace('-', '').replace(' ', '')
        import re
        match = re.match(r'^([A-Z]{2,3})(\d{1,4})([A-Z]?)$', clean_text)
        
        if match:
            letters, numbers, suffix = match.groups()
            formatted = f"{letters}-{numbers}{suffix}"
            return formatted
        
        return text

def is_reasonable_plate_text(text):
    if not text or len(text) < 5 or len(text) > 10:
        return False
    
    has_letters = any(c.isalpha() for c in text)
    has_numbers = any(c.isdigit() for c in text)
    
    if not (has_letters and has_numbers):
        return False
    
    for char in set(text):
        if text.count(char) > len(text) * 0.6:
            return False
    
    return True

def smart_character_ordering(char_boxes, plate_shape):
    if not char_boxes:
        return [], []
    
    plate_height, plate_width = plate_shape[:2]
    row_threshold = plate_height * 0.3
    
    y_positions = [box['y'] + box['h']/2 for box in char_boxes]
    y_positions.sort()
    median_y = y_positions[len(y_positions)//2]
    
    top_row = [box for box in char_boxes if (box['y'] + box['h']/2) < median_y]
    bottom_row = [box for box in char_boxes if (box['y'] + box['h']/2) >= median_y]
    
    is_two_row = (len(top_row) >= 2 and len(bottom_row) >= 2 and 
                  len(top_row) + len(bottom_row) >= 5)
    
    if is_two_row:
        top_row.sort(key=lambda x: x['x'])
        bottom_row.sort(key=lambda x: x['x'])
        
        top_letters = sum(1 for box in top_row if box['is_letter'])
        bottom_numbers = sum(1 for box in bottom_row if not box['is_letter'])
        
        if top_letters >= len(top_row) * 0.6 and bottom_numbers >= len(bottom_row) * 0.6:
            ordered_boxes = top_row + bottom_row
        else:
            char_boxes.sort(key=lambda x: x['x'])
            ordered_boxes = char_boxes
    else:
        char_boxes.sort(key=lambda x: x['x'])
        ordered_boxes = char_boxes
    
    chars = [box['char'] for box in ordered_boxes]
    confidences = [box['conf'] for box in ordered_boxes]
    
    return chars, confidences

def run_enhanced_plate_detection():
    """Enhanced plate detection with tracking and consensus"""
    
    if not os.path.exists(CONFIG['PLATE_MODEL_PATH']) or not os.path.exists(CONFIG['CHAR_MODEL_PATH']):
        print("❌ Model paths are invalid")
        print(f"Plate model: {CONFIG['PLATE_MODEL_PATH']}")
        print(f"Char model: {CONFIG['CHAR_MODEL_PATH']}")
        return

    source = CONFIG['VIDEO_SOURCE']
    if not (os.path.isfile(source) or source.startswith("rtsp://") or source.startswith("http://")):
        print("❌ Video source not found or invalid:", source)
        return

    print("🚗 Enhanced Vehicle License Plate Detection System")
    print("=" * 60)
    print(f"📹 Video Source: {source}")
    print(f"🎯 Plate Model: {os.path.basename(CONFIG['PLATE_MODEL_PATH'])}")
    print(f"🔤 Char Model: {os.path.basename(CONFIG['CHAR_MODEL_PATH'])}")
    print(f"🎛️  Confidence Threshold: {CONFIG['CONFIDENCE_THRESHOLD']}")
    if CONFIG.get('DATABASE_ENABLED'):
        print(f"🗄️  Database: ENABLED | Camera: {CONFIG['DEVICE']} | Direction: {CONFIG['IN_OUT']}")
    else:
        print(f"🗄️  Database: DISABLED")
    print("=" * 60)

    try:
        plate_model = YOLO(CONFIG['PLATE_MODEL_PATH'])
        char_model = YOLO(CONFIG['CHAR_MODEL_PATH'])
        print("✅ Models loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load models: {e}")
        return
    
    db_manager = None
    saved_plates = set()
    detects_dir = os.path.join(SCRIPT_DIR, "detects")
    crops_dir = os.path.join(detects_dir, "crops")
    raw_dir = os.path.join(detects_dir, "raw")
    annotated_dir = os.path.join(detects_dir, "annotated")
    for d in [detects_dir, crops_dir, raw_dir, annotated_dir]:
        if not os.path.exists(d):
            os.makedirs(d)
    if CONFIG.get('DATABASE_ENABLED', False):
        try:
            db_manager = DatabaseManager(
                base_url=CONFIG['API_BASE_URL'],
                cam_code='',
                device=CONFIG['DEVICE'],
                in_out=CONFIG['IN_OUT']
            )
        except Exception as e:
            print(f"⚠️  Database init failed: {e}")
    
    tracker = PlateTracker(
        max_age=CONFIG['TRACKER_MAX_AGE'],
        min_hits=CONFIG['TRACKER_MIN_HITS'],
        iou_threshold=CONFIG['TRACKER_IOU_THRESHOLD']
    )
    validator = SriLankanPlateValidator()

    cap = cv2.VideoCapture(source)
    
    if source.startswith("rtsp://") or source.startswith("http://"):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print("📡 Configured for RTSP stream")
    
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
    frame_count = 0
    start_time = time.time()
    
    cv2.namedWindow("Enhanced License Plate Detection", cv2.WINDOW_NORMAL)
    plate_windows = {}
    
    print("🎮 Controls: 'q' to quit, 's' to save current frame")
    print("-" * 50)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Frame grab failed, retrying...")
            time.sleep(0.1)
            continue

        frame_count += 1
        current_time = time.time()
        
        if source.startswith("rtsp://") or source.startswith("http://"):
            if frame_count % CONFIG['FRAME_SKIP'] != 0:
                continue

        try:
            plate_results = plate_model.predict(source=frame, imgsz=640, conf=CONFIG['CONFIDENCE_THRESHOLD'], verbose=False)
            plate_res = plate_results[0]
        except Exception as e:
            print(f"⚠️ Plate detection error: {e}")
            continue

        annotated_frame = frame.copy()
        detections = []
        
        if len(plate_res.boxes) > 0:
            for i, box in enumerate(plate_res.boxes.xyxy):
                x1, y1, x2, y2 = box.cpu().numpy().astype(int)
                confidence = float(plate_res.boxes.conf[i])
                
                if confidence < CONFIG['CONFIDENCE_THRESHOLD']:
                    continue
                
                plate_height = y2 - y1
                plate_width = x2 - x1
                
                pad_x = int(plate_width * 0.1)
                pad_y = int(plate_height * 0.1)
                x1_crop = max(0, x1 - pad_x)
                y1_crop = max(0, y1 - pad_y)
                x2_crop = min(frame.shape[1], x2 + pad_x)
                y2_crop = min(frame.shape[0], y2 + pad_y)
                
                plate_crop = frame[y1_crop:y2_crop, x1_crop:x2_crop]
                
                if plate_crop.size == 0 or plate_crop.shape[0] < 20 or plate_crop.shape[1] < 50:
                    continue

                temp_crop_path = os.path.join(tempfile.gettempdir(), f"temp_plate_{frame_count}_{i}.jpg")
                cv2.imwrite(temp_crop_path, plate_crop)
                
                try:
                    char_results = char_model.predict(temp_crop_path, imgsz=640, conf=0.3, verbose=False)
                    char_res = char_results[0]

                    plate_text = "No text detected"
                    char_confidence = 0.0
                    
                    if len(char_res.boxes) > 0:
                        char_boxes = []
                        for j, cbox in enumerate(char_res.boxes.xyxy):
                            x1c, y1c, x2c, y2c = cbox.cpu().numpy().astype(int)
                            char_conf = float(char_res.boxes.conf[j])
                            char_cls = int(char_res.boxes.cls[j]) if char_res.boxes.cls is not None else 0
                            
                            if char_conf > 0.3:
                                char = validator.class_to_char.get(char_cls, str(char_cls))
                                char_boxes.append({
                                    'x': x1c, 'y': y1c, 'w': x2c-x1c, 'h': y2c-y1c,
                                    'char': char, 'conf': char_conf, 'is_letter': char.isalpha()
                                })
                        
                        chars, confidences = smart_character_ordering(char_boxes, plate_crop.shape)
                        
                        if chars and len(chars) >= 4:
                            plate_text = "".join(chars)
                            char_confidence = np.mean(confidences) if confidences else 0.0
                            
                            if is_reasonable_plate_text(plate_text):
                                pass
                            else:
                                plate_text = "No text detected"
                                char_confidence = 0.0
                    
                    validated_text, validated_conf, is_valid = validator.validate_and_correct(plate_text, char_confidence)
                    
                    detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'text': validated_text,
                        'confidence': validated_conf,
                        'is_valid': is_valid,
                        'crop': plate_crop
                    })
                    
                except Exception as e:
                    print(f"⚠️ Character recognition error: {e}")
                finally:
                    if os.path.exists(temp_crop_path):
                        os.remove(temp_crop_path)
        
        tracks = tracker.update(detections)
        
        current_track_ids = set(tracks.keys())
        windows_to_remove = []
        for window_name in list(plate_windows.keys()):
            if window_name.startswith("Plate "):
                try:
                    track_id = int(window_name.split(" ")[1])
                    if track_id not in current_track_ids:
                        cv2.destroyWindow(window_name)
                        windows_to_remove.append(window_name)
                        print(f"🗑️  Closed window for disappeared track {track_id}")
                except (ValueError, IndexError):
                    pass
        
        for window_name in windows_to_remove:
            del plate_windows[window_name]
        
        for track_id, track in tracks.items():
            if track['hits'] >= tracker.min_hits:
                _, _, track['is_valid'] = validator.validate_and_correct(track['consensus_text'], track['avg_confidence'])
                
                if (track.get('is_valid', False) and not track.get('saved_to_db', False)
                    and track['avg_confidence'] >= CONFIG['MIN_CONFIDENCE_FOR_DB']
                    and track['hits'] >= CONFIG['MIN_HITS_FOR_DB']):
                    should_save = True
                    plate_text = track['consensus_text']
                    for saved_plate in saved_plates:
                        similarity = tracker.calculate_plate_similarity(plate_text, saved_plate)
                        if similarity > 0.85:
                            should_save = False
                            print(f"🔄 Skipping similar plate: {plate_text} (similar to saved: {saved_plate}, similarity: {similarity:.2f})")
                            break
                    if should_save:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        base_name = f"{track_id}_{plate_text}_{int(track['avg_confidence']*100)}_{timestamp}"
                        # 1. Save plate crop
                        crop_path = os.path.join(crops_dir, f"crop_{base_name}.jpg")
                        cv2.imwrite(crop_path, track['crop'])
                        # 2. Save full frame (no writings or bbox)
                        raw_path = os.path.join(raw_dir, f"raw_{base_name}.jpg")
                        cv2.imwrite(raw_path, frame)
                        # 3. Save full frame (with writings and bbox)
                        annotated = annotated_frame.copy()
                        x1, y1, x2, y2 = track['bbox']
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0,255,0), 2)
                        info_text = f"Plate: {plate_text} | Conf: {track['avg_confidence']:.2f} | Track: {track_id} | Time: {timestamp}"
                        cv2.putText(annotated, info_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                        annotated_path = os.path.join(annotated_dir, f"annotated_{base_name}.jpg")
                        cv2.imwrite(annotated_path, annotated)
                        track['saved_to_db'] = True
                        saved_plates.add(plate_text)
                        print(f"💾 Saved crop: {crop_path}")
                        print(f"💾 Saved raw frame: {raw_path}")
                        print(f"💾 Saved annotated frame: {annotated_path}")
                    else:
                        track['saved_to_db'] = True
        
        valid_tracks = 0
        for track_id, track in tracks.items():
            x1, y1, x2, y2 = track['bbox']
            
            if track['hits'] >= tracker.min_hits:
                color = (0, 255, 0) if track.get('is_valid', False) else (0, 165, 255)
                valid_tracks += 1
            else:
                color = (128, 128, 128)
            
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            db_status = ""
            if (track.get('is_valid', False) and 
                track.get('saved_to_db', False)):
                db_status = " [DB]"
            
            info_text = f"ID:{track_id} {track['consensus_text']} ({track['avg_confidence']:.2f}){db_status}"
            cv2.putText(annotated_frame, info_text, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            hits_text = f"Hits: {track['hits']}"
            cv2.putText(annotated_frame, hits_text, (x1, y2+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            if (track['hits'] >= tracker.min_hits and 
                CONFIG['SHOW_INDIVIDUAL_PLATES'] and 
                'crop' in track):
                
                window_name = f"Plate {track_id}"
                if window_name not in plate_windows:
                    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
                    plate_windows[window_name] = True
                
                crop_resized = cv2.resize(track['crop'], (300, 100), interpolation=cv2.INTER_CUBIC)
                
                text_overlay = crop_resized.copy()
                cv2.putText(text_overlay, track['consensus_text'], (10, 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(text_overlay, f"Conf: {track['avg_confidence']:.2f}", (10, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(text_overlay, f"Hits: {track['hits']}", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(text_overlay, "VALID" if track.get('is_valid', False) else "INVALID", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0) if track.get('is_valid', False) else (0, 0, 255), 1)
                
                cv2.imshow(window_name, text_overlay)

        elapsed_time = current_time - start_time
        current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        
        status_text = f"FPS: {current_fps:.1f} | Frame: {frame_count} | Tracks: {len(tracks)} | Valid: {valid_tracks}"
        cv2.putText(annotated_frame, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated_frame, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

        cv2.imshow("Enhanced License Plate Detection", annotated_frame)
        cv2.resizeWindow("Enhanced License Plate Detection", annotated_frame.shape[1], annotated_frame.shape[0])

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and CONFIG['SAVE_DETECTIONS']:
            timestamp = int(time.time())
            save_path = f"enhanced_detected_plates_{timestamp}.jpg"
            cv2.imwrite(save_path, annotated_frame)
            print(f"💾 Saved frame: {save_path}")

            for track_id, track in tracks.items():
                if 'crop' in track and track['hits'] >= tracker.min_hits:
                    crop_path = f"enhanced_plate_{track_id}_{timestamp}.jpg"
                    cv2.imwrite(crop_path, track['crop'])
                    print(f"💾 Saved plate crop: {crop_path} - '{track['consensus_text']}'")

        if frame_count % 60 == 0 and tracks:
            print(f"🎯 Frame {frame_count}: Tracking {len(tracks)} plates ({valid_tracks} mature)")
            for track_id, track in tracks.items():
                if track['hits'] >= tracker.min_hits:
                    status = "VALID" if track.get('is_valid', False) else "INVALID"
                    print(f"   Track {track_id}: '{track['consensus_text']}' ({status}, {track['avg_confidence']:.2f}, {track['hits']} hits)")
                    
                    if len(track['text_history']) > 1:
                        recent_texts = track['text_history'][-5:]
                        print(f"      Recent detections: {recent_texts}")

    cap.release()
    for window_name in plate_windows:
        cv2.destroyWindow(window_name)
    cv2.destroyAllWindows()
    
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0
    print("\n" + "="*60)
    print("🏁 Enhanced Detection Session Complete")
    print(f"📊 Processed {frame_count} frames in {total_time:.1f} seconds")
    print(f"⚡ Average FPS: {avg_fps:.2f}")
    print(f"🔢 Maximum track ID reached: {tracker.next_id-1}")
    print(f"🎯 Total tracks created: {tracker.next_id-1}")
    print("="*60)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        CONFIG['VIDEO_SOURCE'] = sys.argv[1]
        print(f"📹 Using video source from command line: {CONFIG['VIDEO_SOURCE']}")
    
    run_enhanced_plate_detection()
