from ultralytics import YOLO
import cv2
import os
import time
import numpy as np
from datetime import datetime
from collections import deque
import threading
from queue import Queue

# Dynamic paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, 'models')

# Configuration - easily adjustable
CONFIG = {
    'TRACKER_IOU_THRESHOLD': 0.3,
    'TRACKER_MAX_AGE': 30,
    
    # Model paths
    'PLATE_MODEL_PATH': os.path.join(MODELS_DIR, 'plate_detection.pt'),
    
    # Video source
    'VIDEO_SOURCE': "rtsp://admin:Admin%4001%21@192.168.100.132:554/Streaming/Channels/101",
    
    # Detection settings
    'CONFIDENCE_THRESHOLD': 0.5,
    'FRAME_SKIP': 2,  # Process every Nth frame
    
    # ROI settings (Region of Interest)
    'ROI_X1': 500,
    'ROI_Y1': 400,
    'ROI_X2': 1920,
    'ROI_Y2': 1080,
    'ROI_COLOR': (0, 255, 255),  # Yellow
    'ROI_THICKNESS': 3,
    'USE_ROI_CROP': False,  # Set True to detect only in ROI (faster)
    
    # Dataset capture settings
    'DATASET_FOLDER': 'dataset_frames',
    'DUPLICATE_THRESHOLD': 0.95,
    'MIN_BBOX_AREA': 1000,
    'POSITION_TOLERANCE': 30,
    'MIN_CAPTURE_INTERVAL': 1.0,  # Seconds between captures per track
    'MAX_TRACK_HISTORY': 100,
    
    # Stream optimization
    'RTSP_BUFFER_SIZE': 1,  # Minimal buffering
    'FLUSH_BUFFER_ON_SAVE': True,  # Clear buffer after save
    'THREADED_SAVE': True,  # Save frames in background thread
    'MAX_SAVE_QUEUE': 10,  # Max frames waiting to be saved
    
    # Display settings
    'WINDOW_WIDTH': 1280,
    'WINDOW_HEIGHT': 720,
    'SHOW_STATS': True,
}


class ThreadedFrameSaver:
    """Background thread for non-blocking frame saves"""
    
    def __init__(self, max_queue_size=10):
        self.save_queue = Queue(maxsize=max_queue_size)
        self.running = True
        self.saved_count = 0
        self.dropped_count = 0
        
        # Start saver thread
        self.thread = threading.Thread(target=self._save_worker, daemon=True)
        self.thread.start()
    
    def _save_worker(self):
        """Background worker that saves frames"""
        while self.running:
            try:
                item = self.save_queue.get(timeout=0.1)
                if item is None:  # Poison pill
                    break
                
                filepath, frame = item
                cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                self.saved_count += 1
                self.save_queue.task_done()
                
            except:
                continue
    
    def save_async(self, filepath, frame):
        """Queue frame for saving (non-blocking)"""
        try:
            # Try to add to queue without blocking
            self.save_queue.put_nowait((filepath, frame.copy()))
            return True
        except:
            # Queue full - drop frame
            self.dropped_count += 1
            print(f"⚠️ Save queue full, dropped frame (total dropped: {self.dropped_count})")
            return False
    
    def shutdown(self):
        """Wait for all saves to complete and shutdown"""
        self.save_queue.join()  # Wait for queue to empty
        self.running = False
        self.save_queue.put(None)  # Poison pill
        self.thread.join(timeout=5)
    
    def get_queue_size(self):
        """Get current queue size"""
        return self.save_queue.qsize()


class SimpleIOUTracker:
    """Efficient IOU-based object tracker"""
    def __init__(self, iou_threshold=0.3, max_age=30):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.tracks = {}
        self.next_id = 1
        self.frame_count = 0

    def update(self, detections):
        self.frame_count += 1
        updated_tracks = {}
        assigned = set()
        
        # Match detections to existing tracks
        for track_id, track in self.tracks.items():
            best_match_idx, best_iou = -1, 0
            for i, det in enumerate(detections):
                if i in assigned:
                    continue
                iou = self._calculate_iou(track['bbox'], det['bbox'])
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_match_idx = i
            
            if best_match_idx >= 0:
                det = detections[best_match_idx]
                track['bbox'] = det['bbox']
                track['confidence'] = det['confidence']
                track['last_seen'] = self.frame_count
                updated_tracks[track_id] = track
                assigned.add(best_match_idx)
            elif self.frame_count - track['last_seen'] < self.max_age:
                updated_tracks[track_id] = track
        
        # Add new tracks for unmatched detections
        for i, det in enumerate(detections):
            if i not in assigned:
                updated_tracks[self.next_id] = {
                    'bbox': det['bbox'],
                    'confidence': det['confidence'],
                    'last_seen': self.frame_count,
                    'first_seen': self.frame_count
                }
                self.next_id += 1
        
        self.tracks = updated_tracks
        return self.tracks

    @staticmethod
    def _calculate_iou(box1, box2):
        """Fast IOU calculation"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0


class DatasetCapture:
    """Optimized dataset capture system with non-blocking saves"""
    
    def __init__(self):
        self.capture_count = 0
        self.tracker = SimpleIOUTracker(
            iou_threshold=CONFIG['TRACKER_IOU_THRESHOLD'],
            max_age=CONFIG['TRACKER_MAX_AGE']
        )
        
        # Memory-efficient track history
        self.track_history = {}
        self.max_history = CONFIG['MAX_TRACK_HISTORY']
        
        # Threaded saver for non-blocking I/O
        if CONFIG['THREADED_SAVE']:
            self.frame_saver = ThreadedFrameSaver(max_queue_size=CONFIG['MAX_SAVE_QUEUE'])
        else:
            self.frame_saver = None
        
        # Create dataset directory
        self.detects_dir = os.path.join(SCRIPT_DIR, "detects")
        self.dataset_dir = os.path.join(self.detects_dir, CONFIG['DATASET_FOLDER'])
        os.makedirs(self.dataset_dir, exist_ok=True)
        print(f"📁 Dataset directory: {self.dataset_dir}")
    
    def _get_bbox_center(self, bbox):
        """Calculate bounding box center"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)
    
    def _calculate_perceptual_hash(self, frame, bbox):
        """Fast perceptual hash for duplicate detection"""
        x1, y1, x2, y2 = [int(v) for v in bbox]
        
        # Validate coordinates
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        region = frame[y1:y2, x1:x2]
        
        # Resize to small fixed size and convert to grayscale
        small = cv2.resize(region, (16, 16))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        
        # Create perceptual hash
        avg = gray.mean()
        return (gray > avg).astype(np.uint8).tobytes()
    
    def _calculate_similarity(self, hash1, hash2):
        """Calculate similarity between two hashes"""
        if hash1 is None or hash2 is None:
            return 0.0
        
        # Hamming distance
        diff = sum(b1 != b2 for b1, b2 in zip(hash1, hash2))
        similarity = 1.0 - (diff / len(hash1))
        return similarity
    
    def is_in_roi(self, bbox):
        """Check if bounding box center is inside ROI"""
        center_x, center_y = self._get_bbox_center(bbox)
        
        return (CONFIG['ROI_X1'] <= center_x <= CONFIG['ROI_X2'] and 
                CONFIG['ROI_Y1'] <= center_y <= CONFIG['ROI_Y2'])
    
    def should_capture(self, frame, bbox, track_id, current_time):
        """Determine if frame should be captured"""
        
        # Initialize track history if new
        if track_id not in self.track_history:
            self.track_history[track_id] = deque(maxlen=5)
            return True
        
        history = self.track_history[track_id]
        
        # Check time interval
        if history:
            last_capture = history[-1]
            if current_time - last_capture['time'] < CONFIG['MIN_CAPTURE_INTERVAL']:
                return False
        
        # Check position change
        current_center = self._get_bbox_center(bbox)
        if history:
            last_center = history[-1]['center']
            distance = np.sqrt(
                (current_center[0] - last_center[0])**2 + 
                (current_center[1] - last_center[1])**2
            )
            
            # Significant movement = likely not duplicate
            if distance > CONFIG['POSITION_TOLERANCE'] * 2:
                return True
            
            # Small movement = check similarity
            if distance < CONFIG['POSITION_TOLERANCE']:
                current_hash = self._calculate_perceptual_hash(frame, bbox)
                similarity = self._calculate_similarity(current_hash, last_capture['hash'])
                
                if similarity > CONFIG['DUPLICATE_THRESHOLD']:
                    return False
        
        return True
    
    def capture_frame(self, frame, bbox, confidence, track_id):
        """Capture and save frame (non-blocking if threaded)"""
        current_time = time.time()
        
        if not self.should_capture(frame, bbox, track_id, current_time):
            return False
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"dataset_{self.capture_count:06d}_t{track_id}_c{int(confidence*100)}.jpg"
        filepath = os.path.join(self.dataset_dir, filename)
        
        # Save frame (threaded or blocking)
        if self.frame_saver:
            save_success = self.frame_saver.save_async(filepath, frame)
            if not save_success:
                return False  # Queue full, skip this capture
        else:
            cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        # Store minimal capture info
        current_hash = self._calculate_perceptual_hash(frame, bbox)
        capture_info = {
            'time': current_time,
            'center': self._get_bbox_center(bbox),
            'hash': current_hash,
            'filename': filename
        }
        
        # Update history
        if track_id not in self.track_history:
            self.track_history[track_id] = deque(maxlen=5)
        self.track_history[track_id].append(capture_info)
        
        self.capture_count += 1
        
        queue_info = ""
        if self.frame_saver:
            queue_size = self.frame_saver.get_queue_size()
            if queue_size > 0:
                queue_info = f" [Queue: {queue_size}]"
        
        print(f"📸 Captured #{self.capture_count}: {filename} (t:{track_id}, c:{confidence:.2f}){queue_info}")
        
        return True
    
    def draw_roi(self, frame):
        """Draw ROI rectangle on frame"""
        cv2.rectangle(frame, 
                     (CONFIG['ROI_X1'], CONFIG['ROI_Y1']), 
                     (CONFIG['ROI_X2'], CONFIG['ROI_Y2']), 
                     CONFIG['ROI_COLOR'], 
                     CONFIG['ROI_THICKNESS'])
        cv2.putText(frame, "ROI", 
                   (CONFIG['ROI_X1'], CONFIG['ROI_Y1'] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, CONFIG['ROI_COLOR'], 2)
    
    def cleanup_old_tracks(self):
        """Remove old track history to prevent memory buildup"""
        if len(self.track_history) > CONFIG['MAX_TRACK_HISTORY']:
            active_ids = set(self.tracker.tracks.keys())
            old_ids = [tid for tid in self.track_history.keys() if tid not in active_ids]
            
            if len(old_ids) > 50:
                for tid in old_ids[:-50]:
                    del self.track_history[tid]
    
    def get_stats(self):
        """Get capture statistics"""
        stats = {
            'total_captured': self.capture_count,
            'active_tracks': len(self.tracker.tracks),
            'tracked_plates': len(self.track_history),
            'dataset_folder': self.dataset_dir
        }
        
        if self.frame_saver:
            stats['save_queue'] = self.frame_saver.get_queue_size()
            stats['saved_count'] = self.frame_saver.saved_count
            stats['dropped_count'] = self.frame_saver.dropped_count
        
        return stats
    
    def shutdown(self):
        """Clean shutdown"""
        if self.frame_saver:
            print("💾 Waiting for remaining saves to complete...")
            self.frame_saver.shutdown()
            print(f"✅ All frames saved (dropped: {self.frame_saver.dropped_count})")


def flush_rtsp_buffer(cap, num_frames=5):
    """Flush old frames from RTSP buffer"""
    for _ in range(num_frames):
        cap.grab()  # Discard frame without decoding


def run_dataset_capture():
    """Main dataset capture function with optimized streaming"""
    
    # Validate model
    if not os.path.exists(CONFIG['PLATE_MODEL_PATH']):
        print(f"❌ Model not found: {CONFIG['PLATE_MODEL_PATH']}")
        return
    
    # Validate video source
    source = CONFIG['VIDEO_SOURCE']
    is_stream = source.startswith(("rtsp://", "http://"))
    
    if not is_stream and not os.path.isfile(source):
        print(f"❌ Video file not found: {source}")
        return
    
    print("📊 Dataset Capture System - Stream Optimized")
    print("=" * 60)
    print(f"📹 Source: {source}")
    print(f"🎯 Model: {os.path.basename(CONFIG['PLATE_MODEL_PATH'])}")
    print(f"🎛️  Confidence: {CONFIG['CONFIDENCE_THRESHOLD']}")
    print(f"🔲 ROI: ({CONFIG['ROI_X1']},{CONFIG['ROI_Y1']}) → ({CONFIG['ROI_X2']},{CONFIG['ROI_Y2']})")
    print(f"🚫 Duplicate Threshold: {CONFIG['DUPLICATE_THRESHOLD']}")
    print(f"📏 Position Tolerance: {CONFIG['POSITION_TOLERANCE']}px")
    print(f"🧵 Threaded Saves: {'Enabled' if CONFIG['THREADED_SAVE'] else 'Disabled'}")
    print("=" * 60)
    
    # Load model
    try:
        plate_model = YOLO(CONFIG['PLATE_MODEL_PATH'])
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Initialize systems
    capture_system = DatasetCapture()
    
    # Initialize video capture with optimized settings
    cap = cv2.VideoCapture(source)
    
    if is_stream:
        # Aggressive buffer management for RTSP
        cap.set(cv2.CAP_PROP_BUFFERSIZE, CONFIG['RTSP_BUFFER_SIZE'])
        cap.set(cv2.CAP_PROP_FPS, 25)  # Request lower FPS for stability
        print("📡 Stream mode: Minimal buffering enabled")
    
    # Stats
    frame_count = 0
    processed_count = 0
    start_time = time.time()
    detections_in_roi = 0
    total_detections = 0
    last_flush_time = time.time()
    
    # Create window
    cv2.namedWindow("Dataset Capture", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Dataset Capture", CONFIG['WINDOW_WIDTH'], CONFIG['WINDOW_HEIGHT'])
    
    print("🎮 Controls: 'q'=quit | 's'=save frame | 'r'=show stats | 'f'=flush buffer")
    print("-" * 60)
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                if is_stream:
                    print("⚠️ Stream reconnecting...")
                    time.sleep(0.5)
                    continue
                else:
                    break
            
            frame_count += 1
            
            # Periodic buffer flush for streams (every 5 seconds)
            if is_stream and CONFIG['FLUSH_BUFFER_ON_SAVE']:
                current_time = time.time()
                if current_time - last_flush_time > 5.0:
                    flush_rtsp_buffer(cap, 2)
                    last_flush_time = current_time
            
            # Frame skipping
            if frame_count % CONFIG['FRAME_SKIP'] != 0:
                continue
            
            processed_count += 1
            display_frame = frame.copy()
            capture_system.draw_roi(display_frame)
            
            # Detect plates
            if CONFIG['USE_ROI_CROP']:
                roi_frame = frame[CONFIG['ROI_Y1']:CONFIG['ROI_Y2'], 
                                CONFIG['ROI_X1']:CONFIG['ROI_X2']]
                plate_results = plate_model.predict(source=roi_frame, imgsz=640,
                                                   conf=CONFIG['CONFIDENCE_THRESHOLD'],
                                                   verbose=False)
                offset_x, offset_y = CONFIG['ROI_X1'], CONFIG['ROI_Y1']
            else:
                plate_results = plate_model.predict(source=frame, imgsz=640,
                                                   conf=CONFIG['CONFIDENCE_THRESHOLD'],
                                                   verbose=False)
                offset_x, offset_y = 0, 0
            
            # Process detections
            detections = []
            if len(plate_results[0].boxes) > 0:
                for i, box in enumerate(plate_results[0].boxes.xyxy):
                    x1, y1, x2, y2 = box.cpu().numpy().astype(int)
                    confidence = float(plate_results[0].boxes.conf[i])
                    
                    # Adjust for ROI offset
                    x1, x2 = x1 + offset_x, x2 + offset_x
                    y1, y2 = y1 + offset_y, y2 + offset_y
                    
                    bbox = (x1, y1, x2, y2)
                    bbox_area = (x2 - x1) * (y2 - y1)
                    
                    if bbox_area >= CONFIG['MIN_BBOX_AREA']:
                        detections.append({'bbox': bbox, 'confidence': confidence})
                        total_detections += 1
            
            # Update tracker
            tracks = capture_system.tracker.update(detections)
            
            # Process tracks
            for track_id, track in tracks.items():
                bbox = track['bbox']
                confidence = track['confidence']
                x1, y1, x2, y2 = bbox
                
                in_roi = capture_system.is_in_roi(bbox)
                
                if in_roi:
                    detections_in_roi += 1
                    was_captured = capture_system.capture_frame(frame, bbox, confidence, track_id)
                    
                    # Flush buffer after capture to prevent lag
                    if was_captured and is_stream and CONFIG['FLUSH_BUFFER_ON_SAVE']:
                        flush_rtsp_buffer(cap, 1)
                    
                    color = (0, 255, 0) if was_captured else (0, 255, 255)
                    status = "✓ SAVED" if was_captured else "⊗ DUP"
                else:
                    color = (0, 0, 255)
                    status = "OUT"
                
                # Draw bbox
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display_frame, f"{status} T{track_id}", 
                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(display_frame, f"{confidence:.2f}", 
                           (x1, y2+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
            
            # Stats overlay
            if CONFIG['SHOW_STATS']:
                elapsed = time.time() - start_time
                fps = processed_count / elapsed if elapsed > 0 else 0
                stats = capture_system.get_stats()
                
                overlay = display_frame.copy()
                height = 185 if 'save_queue' in stats else 160
                cv2.rectangle(overlay, (10, 10), (450, height), (0, 0, 0), -1)
                display_frame = cv2.addWeighted(display_frame, 0.7, overlay, 0.3, 0)
                
                info = [
                    f"FPS: {fps:.1f} | Frames: {frame_count} ({processed_count} proc)",
                    f"Detections: {total_detections} | In ROI: {detections_in_roi}",
                    f"Captured: {stats['total_captured']} | Active: {stats['active_tracks']}",
                    f"Tracked: {stats['tracked_plates']} plates"
                ]
                
                if 'save_queue' in stats:
                    info.append(f"Save Queue: {stats['save_queue']} | Dropped: {stats['dropped_count']}")
                
                info.append(f"Uptime: {elapsed:.0f}s")
                
                for i, text in enumerate(info):
                    cv2.putText(display_frame, text, (15, 35 + i*25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
            
            cv2.imshow("Dataset Capture", display_frame)
            
            # Keyboard controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                ts = int(time.time())
                path = os.path.join(capture_system.dataset_dir, f"manual_{ts}.jpg")
                cv2.imwrite(path, frame)
                print(f"💾 Manual save: {path}")
            elif key == ord('f'):
                # Manual buffer flush
                if is_stream:
                    flush_rtsp_buffer(cap, 5)
                    print("🔄 Buffer flushed")
            elif key == ord('r'):
                stats = capture_system.get_stats()
                print(f"\n{'='*50}")
                print(f"📊 Stats | Captured: {stats['total_captured']} | "
                      f"Active: {stats['active_tracks']} | Total: {stats['tracked_plates']}")
                if 'save_queue' in stats:
                    print(f"💾 Queue: {stats['save_queue']} | Saved: {stats['saved_count']} | "
                          f"Dropped: {stats['dropped_count']}")
                print(f"{'='*50}\n")
            
            # Periodic cleanup
            if processed_count % 100 == 0:
                capture_system.cleanup_old_tracks()
    
    finally:
        # Cleanup
        print("\n🛑 Shutting down...")
        capture_system.shutdown()
        cap.release()
        cv2.destroyAllWindows()
    
    # Final report
    elapsed = time.time() - start_time
    stats = capture_system.get_stats()
    
    print(f"\n{'='*60}")
    print("🏁 Session Complete")
    print(f"⏱️  Duration: {elapsed:.1f}s | FPS: {processed_count/elapsed:.1f}")
    print(f"🎯 Detections: {total_detections} total | {detections_in_roi} in ROI")
    print(f"📸 Captured: {stats['total_captured']} frames")
    if 'dropped_count' in stats:
        print(f"⚠️  Dropped: {stats['dropped_count']} frames (queue full)")
    print(f"💾 Location: {stats['dataset_folder']}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import sys
    
    # CLI overrides
    if len(sys.argv) > 1:
        CONFIG['VIDEO_SOURCE'] = sys.argv[1]
        print(f"📹 CLI source: {CONFIG['VIDEO_SOURCE']}")
    
    if len(sys.argv) >= 6:
        try:
            CONFIG['ROI_X1'] = int(sys.argv[2])
            CONFIG['ROI_Y1'] = int(sys.argv[3])
            CONFIG['ROI_X2'] = int(sys.argv[4])
            CONFIG['ROI_Y2'] = int(sys.argv[5])
            print(f"🔲 CLI ROI: ({CONFIG['ROI_X1']},{CONFIG['ROI_Y1']}) → "
                  f"({CONFIG['ROI_X2']},{CONFIG['ROI_Y2']})")
        except ValueError:
            print("⚠️ Invalid ROI, using defaults")
    
    run_dataset_capture()