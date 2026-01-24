"""
Script to capture screenshots of card ROIs only.
Saves player and community card regions to files when triggered.
"""

import cv2
import os
from datetime import datetime
from pathlib import Path
from capture.screen_capture import ScreenCapture
from vision.table_detector import TableDetector
import config


class CardROICapture:
    """Captures and saves card ROI screenshots on demand."""
    
    def __init__(self, output_dir: str = "captured_cards"):
        """
        Initialize the card ROI capture system.
        
        Args:
            output_dir: Directory to save captured card ROIs
        """
        self.output_dir = output_dir
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize capture and detector
        self.cap = ScreenCapture(
            fps=config.CAPTURE_FPS,
            region=config.CAPTURE_REGION,
            output_color="BGR"
        )
        self.detector = TableDetector(
            model_path=config.MODEL_TABLE_PATH,
            conf_thres=config.CONF_THRES,
            device=config.DEVICE
        )
        
        print(f"Card ROI Capture initialized. Output directory: {self.output_dir}")
    
    def capture_and_save_rois(self):
        """
        Capture a single frame and save all card ROIs to separate files.
        
        Returns:
            dict: Information about saved files
        """
        frame = self.cap.get_frame()
        if frame is None:
            print("Failed to capture frame")
            return None
        
        # Detect tables
        tables = self.detector.detect(frame)
        if not tables:
            print("No tables detected")
            return None
        
        # Create timestamped subdirectory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        capture_dir = os.path.join(self.output_dir, timestamp)
        Path(capture_dir).mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        # Process each table
        for table_idx, table in enumerate(tables):
            if table is None or table.w <= 0 or table.h <= 0:
                continue
            
            # Extract and save player and community card ROIs
            for roi_name, (x_pct, y_pct, w_pct, h_pct) in config.TABLE_ROIS.items():
                if "card" in roi_name:
                    x1, y1, x2, y2 = table.roi_from_rel(
                        x_pct=x_pct, y_pct=y_pct, 
                        w_pct=w_pct, h_pct=h_pct
                    )
                    
                    # Validate coordinates
                    if x1 >= 0 and y1 >= 0 and x2 <= frame.shape[1] and y2 <= frame.shape[0]:
                        # Extract ROI
                        card_region = frame[y1:y2, x1:x2]
                        
                        if card_region.size > 0:
                            # Save to file
                            filename = f"{roi_name}_table{table_idx}.png"
                            filepath = os.path.join(capture_dir, filename)
                            cv2.imwrite(filepath, card_region)
                            saved_files[roi_name] = filepath
                            print(f"  Saved: {filename}")
        
        if saved_files:
            print(f"✓ Captured {len(saved_files)} card ROIs to {capture_dir}")
            return {
                "timestamp": timestamp,
                "directory": capture_dir,
                "files": saved_files,
                "count": len(saved_files)
            }
        else:
            print("No card ROIs captured")
            return None
    
    def close(self):
        """Close the capture device."""
        self.cap.stop()
        print("Capture closed")


def main():
    """Interactive mode for capturing card ROIs."""
    capturer = CardROICapture()
    
    print("\n" + "="*50)
    print("Card ROI Capture Tool")
    print("="*50)
    print("Commands:")
    print("  c - Capture card ROIs")
    print("  q - Quit")
    print("="*50 + "\n")
    
    try:
        while True:
            cmd = input("Enter command (c/q): ").strip().lower()
            
            if cmd == "c":
                print("\nCapturing card ROIs...")
                result = capturer.capture_and_save_rois()
                if result:
                    print(f"Saved to: {result['directory']}")
                print()
            
            elif cmd == "q":
                print("Exiting...")
                break
            
            else:
                print("Invalid command. Use 'c' or 'q'")
    
    finally:
        capturer.close()


if __name__ == "__main__":
    main()
