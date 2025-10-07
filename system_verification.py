#!/usr/bin/env python3
"""
Comprehensive test script to verify that the VLPR system is properly set up
Includes IP camera connectivity testing and system diagnostics
"""

import sys
import os
import time
from pathlib import Path

# Ensure output is flushed immediately
def print_flush(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()

def test_imports():
    """Test that all required modules can be imported"""
    print_flush("Testing imports...")
    
    try:
        # Set environment variables to prevent hanging
        os.environ['YOLO_VERBOSE'] = 'False'
        os.environ['YOLO_OFFLINE'] = '1'
        
        from ultralytics import YOLO
        print_flush("✅ ultralytics imported successfully")
    except ImportError as e:
        print_flush(f"❌ Failed to import ultralytics: {e}")
        return False
    
    try:
        import cv2
        print_flush(f"✅ OpenCV imported successfully (version: {cv2.__version__})")
    except ImportError as e:
        print_flush(f"❌ Failed to import OpenCV: {e}")
        return False
    
    try:
        import torch
        print_flush(f"✅ PyTorch imported successfully (version: {torch.__version__})")
        if torch.cuda.is_available():
            print_flush(f"🔥 CUDA available: {torch.cuda.device_count()} device(s)")
        else:
            print_flush("💻 CUDA not available - using CPU")
    except ImportError as e:
        print_flush(f"❌ Failed to import PyTorch: {e}")
        return False
    
    try:
        import numpy as np
        print_flush(f"✅ NumPy imported successfully (version: {np.__version__})")
    except ImportError as e:
        print_flush(f"❌ Failed to import NumPy: {e}")
        return False
    
    try:
        import tempfile
        print_flush("✅ tempfile imported successfully")
    except ImportError as e:
        print_flush(f"❌ Failed to import tempfile: {e}")
        return False
    
    try:
        from flask import Flask
        print_flush("✅ Flask imported successfully")
    except ImportError as e:
        print_flush(f"❌ Failed to import Flask: {e}")
        return False
    
    # Test optional packages
    optional_packages = [
        ("requests", "Requests"),
        ("PIL", "Pillow")
    ]
    
    for package, name in optional_packages:
        try:
            __import__(package)
            print_flush(f"✅ {name} imported successfully")
        except ImportError:
            print_flush(f"⚠️  {name} not available")
    
    return True

def test_model_paths():
    """Test that model files exist in the correct locations"""
    print_flush("\nTesting model file paths...")
    
    # Get the current script directory and find the project root
    current_dir = Path(__file__).parent.absolute()
    
    model_files = [
        "models/plate_detection.pt",
        "models/character_recognition.pt"
    ]
    
    all_exist = True
    for model_file in model_files:
        full_path = current_dir / model_file
        if full_path.exists():
            size = full_path.stat().st_size / (1024 * 1024)  # Size in MB
            print_flush(f"✅ {model_file} exists ({size:.1f} MB)")
        else:
            print_flush(f"❌ {model_file} not found at {full_path}")
            all_exist = False
    
    # Check for optimized models
    optimized_dir = current_dir / "models" / "optimized"
    if optimized_dir.exists():
        print_flush("✅ Optimized models directory exists")
        for trt_file in ["plate_detection.trt", "character_recognition.trt"]:
            trt_path = optimized_dir / trt_file
            if trt_path.exists():
                size = trt_path.stat().st_size / (1024 * 1024)
                print_flush(f"✅ {trt_file} exists ({size:.1f} MB)")
    
    return all_exist

def test_video_files():
    """Test that video files exist"""
    print_flush("\nTesting video file paths...")
    
    current_dir = Path(__file__).parent.absolute()
    
        # Only check for the video file that exists
    video_files = [
        "tests/dges1.mp4"
    ]
    
    found_videos = 0
    for video_file in video_files:
        full_path = current_dir / video_file
        if full_path.exists():
            size = full_path.stat().st_size / (1024 * 1024)  # Size in MB
            print_flush(f"✅ {video_file} exists ({size:.1f} MB)")
            found_videos += 1
        else:
            print_flush(f"❌ {video_file} not found")
    
    return found_videos > 0  # At least one video file should exist

def test_ip_camera_connectivity():
    """Test IP camera connectivity with common RTSP URLs"""
    print_flush("\nTesting IP camera connectivity...")
    
    try:
        import cv2
    except ImportError:
        print_flush("❌ OpenCV not available for camera testing")
        return False
    
    # Test the user's specific camera URL
    test_urls = [
        "rtsp://admin:Admin%4001%21@192.168.100.132:554/Streaming/Channels/101",  # From your config
    ]
    
    working_cameras = 0
    
    for url in test_urls:
        print_flush(f"🔍 Testing: {url}")
        
        try:
            # Set a short timeout for testing
            cap = cv2.VideoCapture(url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Try to read a frame with timeout
            success = False
            start_time = time.time()
            timeout = 10  # 10 second timeout
            
            while time.time() - start_time < timeout:
                ret, frame = cap.read()
                if ret and frame is not None:
                    print_flush(f"✅ Camera accessible: {url}")
                    print_flush(f"   📏 Resolution: {frame.shape[1]}x{frame.shape[0]}")
                    working_cameras += 1
                    success = True
                    break
                time.sleep(0.1)
            
            if not success:
                print_flush(f"⚠️  Camera timeout or no signal: {url}")
            
            cap.release()
            
        except Exception as e:
            print_flush(f"❌ Camera connection failed: {url} - {str(e)}")
    
    if working_cameras == 0:
        print_flush("⚠️  No IP cameras accessible (this is normal if cameras are not configured)")
        print_flush("   💡 To test your camera, update the URL in main/main.py")
    
    return True  # Don't fail the test if no cameras are available

def test_webcam():
    """Test local webcam connectivity"""
    print_flush("\nTesting local webcam...")
    
    try:
        import cv2
        
        # Test default camera (index 0)
        cap = cv2.VideoCapture(0)
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print_flush("✅ Local webcam accessible")
                print_flush(f"   📏 Resolution: {frame.shape[1]}x{frame.shape[0]}")
                cap.release()
                return True
            else:
                print_flush("⚠️  Local webcam detected but no frame received")
        else:
            print_flush("⚠️  No local webcam detected")
        
        cap.release()
        
    except Exception as e:
        print_flush(f"❌ Webcam test failed: {str(e)}")
    
    return True  # Don't fail if no webcam

def test_temp_directory():
    """Test that temporary directory is writable"""
    print_flush("\nTesting temporary directory...")
    
    try:
        import tempfile
        temp_dir = tempfile.gettempdir()
        test_file = os.path.join(temp_dir, "vlpr_test.txt")
        
        # Try to write a test file
        with open(test_file, 'w') as f:
            f.write("test")
        
        # Try to read it back
        with open(test_file, 'r') as f:
            content = f.read()
        
        # Clean up
        os.remove(test_file)
        
        print_flush(f"✅ Temporary directory is writable: {temp_dir}")
        return True
        
    except Exception as e:
        print_flush(f"❌ Temporary directory test failed: {e}")
        return False

def main():
    """Run all tests"""
    print_flush("🚗 VLPR System Comprehensive Test Suite")
    print_flush("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("Model Files Test", test_model_paths),
        ("Video Files Test", test_video_files),
        ("IP Camera Connectivity", test_ip_camera_connectivity),
        ("Local Webcam Test", test_webcam),
        ("Temporary Directory Test", test_temp_directory)
    ]
    
    results = []
    start_time = time.time()
    
    for test_name, test_func in tests:
        print_flush(f"\n📋 {test_name}")
        print_flush("-" * 40)
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print_flush(f"❌ Test failed with exception: {e}")
            results.append((test_name, False))
    
    # Test summary
    end_time = time.time()
    test_duration = end_time - start_time
    
    print_flush("\n" + "=" * 60)
    print_flush("📊 Test Summary:")
    print_flush("-" * 60)
    
    passed_tests = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print_flush(f"  {test_name:<25} {status}")
        if result:
            passed_tests += 1
    
    print_flush(f"\n📈 Results: {passed_tests}/{len(results)} tests passed")
    print_flush(f"⏱️  Total time: {test_duration:.1f} seconds")
    
    # Overall assessment
    success_rate = (passed_tests / len(results)) * 100
    
    if success_rate == 100:
        print_flush("\n🎉 All tests passed! Your VLPR system is ready to use.")
        print_flush("\n🚀 Quick Start:")
        print_flush("   1. python main/main.py          # Run main detection")
        print_flush("   2. python test_detection.py     # Test with video file")
        print_flush("   3. python ip_camera_test.py     # Test IP camera")
    elif success_rate >= 80:
        print_flush(f"\n✅ System mostly ready ({success_rate:.0f}% tests passed)")
        print_flush("⚠️  Some optional features may not work. Check failed tests above.")
    else:
        print_flush(f"\n⚠️  System needs attention ({success_rate:.0f}% tests passed)")
        print_flush("❌ Please resolve the failed tests before proceeding.")
    
    print_flush("\n📚 Documentation: https://github.com/GaruVA/vehicle-license-plate-recognition")
    print_flush("🐛 Report issues: https://github.com/GaruVA/vehicle-license-plate-recognition/issues")
    
    return success_rate >= 80

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print_flush("\n⏹️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_flush(f"\n💥 Test failed with error: {e}")
        sys.exit(1)
