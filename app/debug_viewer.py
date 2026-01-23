import time
import cv2
import numpy as np


class DebugViewer:
    def __init__(self, window_name: str):
        self.window_name = window_name
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        
        # Create separate debug output window
        self.debug_window_name = "Debug Output"
        cv2.namedWindow(self.debug_window_name, cv2.WINDOW_NORMAL)
        
        self._last_t = time.time()
        self._frames = 0
        
        # Store debug messages for display
        self.debug_messages = []
        self.max_messages = 20  # Keep last 20 messages

    def show(self, frame, detected_cards: dict = None):
        """
        Display frame with optional detected cards in separate debug window.
        
        Args:
            frame: Frame to display
            detected_cards: Dictionary of detected cards to display in debug window
                           Format: {"player_card_1": "As", "community_card_1": "Kh", ...}
        """
        cv2.imshow(self.window_name, frame)
        
        # Show debug window
        if detected_cards is not None:
            self._update_debug_window(detected_cards)
        
        key = cv2.waitKey(1) & 0xFF
        return key
    
    def add_debug_message(self, message: str):
        """Add a message to the debug output."""
        timestamp = time.strftime("%H:%M:%S")
        self.debug_messages.append(f"[{timestamp}] {message}")
        
        # Keep only last N messages
        if len(self.debug_messages) > self.max_messages:
            self.debug_messages.pop(0)
    
    def _update_debug_window(self, detected_cards: dict):
        """
        Update the debug output window with detected cards info.
        
        Args:
            detected_cards: Dictionary of detected cards
        """
        from vision.card_detector import CardClassifier
        
        # Create debug panel
        panel_width = 400
        panel_height = 600
        panel = np.ones((panel_height, panel_width, 3), dtype=np.uint8) * 25
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        text_color = (0, 255, 0)  # Green text
        header_color = (0, 255, 255)  # Cyan for headers
        
        y_position = 20
        line_height = 20
        padding = 10
        
        # Title
        cv2.putText(panel, "=== CARD DETECTION DEBUG ===", (padding, y_position),
                   font, font_scale + 0.1, header_color, font_thickness + 1, cv2.LINE_AA)
        y_position += line_height + 5
        
        # Draw separator line
        cv2.line(panel, (padding, y_position), (panel_width - padding, y_position), (100, 100, 100), 1)
        y_position += 10
        
        # Display detected cards
        if not detected_cards:
            cv2.putText(panel, ">> No cards detected", (padding, y_position),
                       font, font_scale, (0, 0, 255), font_thickness, cv2.LINE_AA)  # Red for no detection
            y_position += line_height
        else:
            # Organize cards by type
            player_cards = {}
            community_cards = {}
            
            for roi_name, card in detected_cards.items():
                if "player_card" in roi_name:
                    player_cards[roi_name] = card
                elif "community_card" in roi_name:
                    community_cards[roi_name] = card
            
            if player_cards:
                cv2.putText(panel, "PLAYER CARDS:", (padding, y_position),
                           font, font_scale, header_color, font_thickness, cv2.LINE_AA)
                y_position += line_height
                
                for roi_name in sorted(player_cards.keys()):
                    card = player_cards[roi_name]
                    label = card.get("label", "?")
                    conf = card.get("card_conf", 0.0)
                    text = f"  {roi_name}: {label} ({conf:.2f})"
                    cv2.putText(panel, text, (padding, y_position),
                               font, font_scale, text_color, font_thickness, cv2.LINE_AA)
                    y_position += line_height
            
            if community_cards:
                cv2.putText(panel, "COMMUNITY CARDS:", (padding, y_position),
                           font, font_scale, header_color, font_thickness, cv2.LINE_AA)
                y_position += line_height
                
                for i, (roi_name, card) in enumerate(sorted(community_cards.items()), 1):
                    label = card.get("label", "?")
                    conf = card.get("card_conf", 0.0)
                    text = f"  {i}. {label} ({conf:.2f})"
                    cv2.putText(panel, text, (padding, y_position),
                               font, font_scale, text_color, font_thickness, cv2.LINE_AA)
                    y_position += line_height
        
        y_position += 10
        
        # Draw separator line
        cv2.line(panel, (padding, y_position), (panel_width - padding, y_position), (100, 100, 100), 1)
        y_position += 10
        
        # Display recent debug messages
        cv2.putText(panel, "RECENT EVENTS:", (padding, y_position),
                   font, font_scale, header_color, font_thickness, cv2.LINE_AA)
        y_position += line_height
        
        for msg in self.debug_messages[-8:]:  # Show last 8 messages
            if y_position >= panel_height - 20:
                break
            cv2.putText(panel, msg, (padding, y_position),
                       font, font_scale - 0.1, (200, 200, 200), 1, cv2.LINE_AA)
            y_position += line_height - 5
        
        # Display panel
        cv2.imshow(self.debug_window_name, panel)

    def log_fps(self, prefix="Live FPS"):
        self._frames += 1
        now = time.time()
        if now - self._last_t >= 1.0:
            fps = self._frames / (now - self._last_t)
            print(f"{prefix}: {fps:.1f}")
            self.add_debug_message(f"{prefix}: {fps:.1f}")
            self._last_t = now
            self._frames = 0

    def close(self):
        cv2.destroyAllWindows()
