"""
Game interface for Elden Ring
Handles:
- Capturing game state (screenshots, processing)
- Sending inputs to the game
- Extracting relevant information (health, stamina, stats)

Uses Elden Ring default keyboard/mouse control scheme
"""

import numpy as np
from PIL import ImageGrab
import cv2
from pynput.mouse import Controller as MouseController
from pynput.mouse import Button
from pynput.keyboard import Key, Controller
import time


class GameInterface:
    """Interface between AI and Elden Ring"""
    
    # Elden Ring default control scheme
    CONTROLS = {
        # Movement
        'forward': 'w',
        'backward': 's',
        'left': 'a',
        'right': 'd',
        'backstep': Key.space,
        'jump': 'f',
        'crouch': 'x',
        'walk': Key.alt,
        
        # Combat
        'normal_attack': Button.left,
        'heavy_attack': Key.shift,  # Shift + Left Click
        'guard': Button.right,
        'skill': Key.shift,  # Shift + Right Click
        'use_item': 'r',
        'two_hand': 'e',  # Hold E + Left Click
        
        # Camera & Lock-on
        'lock_on': 'q',
        'reset_camera': Button.middle,
        
        # Menu & Items
        'main_menu': Key.esc,
        'map': 'g',
        'switch_sorcery': Key.up,
        'switch_item': Key.down,
        'switch_right_hand': Key.right,
        'switch_left_hand': Key.left,
        'use_pouch_item': 'e',
        'event_action': 'e',
    }
    
    def __init__(self, window_rect=(0, 0, 1920, 1080)):
        """
        Initialize game interface
        
        Args:
            window_rect: (x1, y1, x2, y2) - coordinates of game window
                        Defaults to (0, 0, 1920, 1080) for full-screen 1080p
                        If game window is elsewhere, adjust these coordinates
        """
        self.window_rect = window_rect
        self.keyboard = Controller()
        self.mouse = MouseController()
        self.last_action_time = time.time()
        
        # Track quickslot inventory state
        # Each slot can be empty (False) or have an item (True)
        self.quickslots = [False] * 8  # 8 quickslots (0-7)
        self.quickslot_history = []  # Track changes over time
        
        # Track health for damage detection
        self.previous_health = 1.0  # Assume full health at start
        self.recent_damage_frames = 0  # Frames since last damage taken (for combat detection)
        
        # Validate window rect on init
        self._validate_window_rect()
    
    def _validate_window_rect(self):
        """Validate that window_rect coordinates are valid"""
        x1, y1, x2, y2 = self.window_rect
        
        # Check for invalid coordinates
        if x1 >= x2 or y1 >= y2:
            raise ValueError(f"Invalid window_rect: {self.window_rect}. Must have x1 < x2 and y1 < y2")
        
        if x1 < 0 or y1 < 0:
            raise ValueError(f"Invalid window_rect: {self.window_rect}. Coordinates cannot be negative")
        
        # Get screen size to validate bounds
        try:
            test_screen = ImageGrab.grab(bbox=self.window_rect)
            if test_screen.size == 0:
                raise ValueError("Screenshot failed: window_rect may be out of bounds")
        except Exception as e:
            raise ValueError(f"Invalid window_rect {self.window_rect}: {str(e)}")
    
    def capture_screen(self):
        """
        Capture current game screen
        
        Returns:
            numpy array of screenshot (H, W, C)
            
        Raises:
            RuntimeError: If screenshot capture fails
        """
        try:
            screenshot = ImageGrab.grab(bbox=self.window_rect)
            if screenshot is None or screenshot.size == 0:
                raise RuntimeError("Screenshot capture returned empty image")
            return np.array(screenshot)
        except Exception as e:
            raise RuntimeError(f"Failed to capture screen with window_rect {self.window_rect}: {str(e)}")
    
    
    def process_screen(self, screen):
        """
        Process raw screenshot for AI input
        Converts to grayscale, resizes, normalizes
        
        Args:
            screen: raw screenshot
            
        Returns:
            processed image (1, 160, 160) - channels first format for CNN
        """
        # Convert to grayscale
        gray = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
        
        # Resize to 160x160 - provides good detail from high-res captures (1080p+)
        resized = cv2.resize(gray, (160, 160))
        
        # Normalize to 0-1
        normalized = resized.astype(np.float32) / 255.0
        
        return np.expand_dims(normalized, axis=0)  # Add channel dimension at start
    
    def get_game_state(self):
        """
        Get current game state including detected health/stamina/doors/quickslots
        
        Returns:
            dict with relevant game information including health percentage and exit types
        """
        screen = self.capture_screen()
        processed_screen = self.process_screen(screen)
        
        # Detect health bar (typically bottom-left of screen in Elden Ring)
        health_percent = self._detect_health_bar(screen)
        stamina_percent = self._detect_stamina_bar(screen)
        exits = self._detect_doors_or_exits(screen)
        
        # Track damage taken this frame
        health_dropped = False
        if health_percent >= 0:  # Valid health detection
            if health_percent < self.previous_health - 0.01:  # Health decreased (>1% drop)
                health_dropped = True
                self.recent_damage_frames = 0  # Reset damage timer
            self.previous_health = health_percent
        
        # Increment damage timer (frames since last damage)
        if self.recent_damage_frames < 120:  # Track for up to 120 frames (~2 seconds)
            self.recent_damage_frames += 1
        
        # Detect quickslot inventory (which slots have items)
        quickslots = self._detect_quickslots(screen)
        self.quickslots = quickslots  # Store for reward calculation
        
        # Detect if inventory screen is open and search for Wizened Finger
        has_wizened_finger = self._detect_wizened_finger(screen)
        
        # Detect if currently in outdoor area (for Chapel exit bonus)
        is_outdoor = self._detect_outdoor_area(screen)
        
        # Detect boss health bar (red bar at bottom during boss fight)
        boss_health_visible = self._detect_boss_health_bar(screen)
        
        # Detect golden fog wall (boss arena barrier)
        fog_wall_visible = self._detect_fog_wall(screen)
        
        # Detect roof/ceiling (indicates indoors)
        roof_visible = self._detect_roof(screen)
        
        # Detect status buildup bar (poison, rot, bleed, etc.)
        status_buildup_visible = self._detect_status_buildup(screen)
        
        # Detect ground items (glowing loot on corpses/ground)
        ground_items_visible = self._detect_ground_items(screen)
        
        # Detect door state (closed door with interaction prompt)
        door_state = self._detect_door_state(screen)
        
        # Detect if in combat
        in_combat = self._detect_in_combat(screen, health_dropped=health_dropped, status_buildup_visible=status_buildup_visible)
        
        return {
            'screen': processed_screen,
            'raw_screen': screen,
            'timestamp': time.time(),
            'health_percent': health_percent,
            'stamina_percent': stamina_percent,
            'exits': exits,  # dict with closed_doors, open_doors, archways, total
            'quickslots': quickslots,  # list of 8 bools: which slots have items
            'has_wizened_finger': has_wizened_finger,  # bool: True if Wizened Finger detected in inventory
            'is_outdoor': is_outdoor,  # bool: True if in outdoor area
            'boss_health_visible': boss_health_visible,  # bool: True if boss health bar visible
            'fog_wall_visible': fog_wall_visible,  # bool: True if golden fog wall visible
            'roof_visible': roof_visible,  # bool: True if roof/ceiling detected
            'status_buildup_visible': status_buildup_visible,  # bool: True if status effect buildup bar visible
            'ground_items_visible': ground_items_visible,  # bool: True if items visible on ground/corpses
            'door_state': door_state,  # dict: door state info (has_closable_door, has_open_prompt)
            'in_combat': in_combat  # bool: True if currently in combat
        }
    
    def _detect_health_bar(self, screen):
        """
        Detect current health percentage from health bar
        Health bar is typically red, located bottom-left of screen
        
        Args:
            screen: raw screenshot
            
        Returns:
            float: health percentage (0.0 to 1.0), or -1 if detection fails
        """
        try:
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(screen, cv2.COLOR_RGB2HSV)
            
            # Red color range for health bar (in HSV)
            # Lower red range
            lower_red1 = np.array([0, 50, 50])
            upper_red1 = np.array([10, 255, 255])
            # Upper red range
            lower_red2 = np.array([170, 50, 50])
            upper_red2 = np.array([180, 255, 255])
            
            # Create masks
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask = mask1 | mask2
            
            # Health bar is typically in bottom-left area
            # Focus on lower 20% and left 30% of screen
            h, w = mask.shape
            roi = mask[int(h * 0.75):h, 0:int(w * 0.25)]
            
            if roi.size == 0 or np.sum(roi) < 100:
                return -1  # No health bar detected
            
            # Find contours (the health bar)
            contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return -1
            
            # Get the largest contour (should be the health bar)
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Approximate health based on bar width
            # Assume full bar is roughly 150 pixels wide (adjust if needed)
            max_bar_width = 150
            health_percent = w / max_bar_width
            
            # CRITICAL: Clamp to valid range [0.0, 1.0]
            health_percent = max(0.0, min(health_percent, 1.0))
            
            return health_percent
        except Exception as e:
            # If detection fails, return -1 (unknown)
            return -1
    
    def _detect_stamina_bar(self, screen):
        """
        Detect current stamina percentage from stamina bar
        Stamina bar is typically green/yellow, located bottom-left below health
        
        Args:
            screen: raw screenshot
            
        Returns:
            float: stamina percentage (0.0 to 1.0), or -1 if detection fails
        """
        try:
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(screen, cv2.COLOR_RGB2HSV)
            
            # Yellow/Green color range for stamina bar (in HSV)
            lower_stamina = np.array([15, 100, 100])
            upper_stamina = np.array([35, 255, 255])
            
            mask = cv2.inRange(hsv, lower_stamina, upper_stamina)
            
            # Stamina bar is typically in bottom-left area, below health
            # Focus on lower 20% and left 30% of screen, but slightly lower than health
            h, w = mask.shape
            roi = mask[int(h * 0.80):h, 0:int(w * 0.25)]
            
            if roi.size == 0 or np.sum(roi) < 100:
                return -1  # No stamina bar detected
            
            # Find contours (the stamina bar)
            contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return -1
            
            # Get the largest contour (should be the stamina bar)
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Approximate stamina based on bar width
            # Assume full bar is roughly 150 pixels wide (adjust if needed)
            max_bar_width = 150
            stamina_percent = w / max_bar_width
            
            # CRITICAL: Clamp to valid range [0.0, 1.0]
            stamina_percent = max(0.0, min(stamina_percent, 1.0))
            
            return stamina_percent
        except Exception as e:
            # If detection fails, return -1 (unknown)
            return -1
    
    def _detect_doors_or_exits(self, screen):
        """
        Detect different types of exits/doors in the game world using ROI-based processing.
        Only processes likely exit regions (edges and center) to reduce noise and computation.
        
        Returns dict with counts of each type
        """
        try:
            h, w = screen.shape[:2]
            
            # Define Regions of Interest (ROI) - exits typically appear here
            rois = [
                ("left_edge", 0, int(w * 0.2), 0, h),           # Left 20%
                ("right_edge", int(w * 0.8), w, 0, h),          # Right 20%
                ("top_edge", 0, w, 0, int(h * 0.1)),            # Top 10%
                ("center", int(w * 0.35), int(w * 0.65), int(h * 0.2), int(h * 0.8))  # Center 30% horizontally
            ]
            
            gray = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
            hsv = cv2.cvtColor(screen, cv2.COLOR_RGB2HSV)
            
            doorway_count = 0
            bright_count = 0
            dark_exits = 0
            
            # Process each ROI
            for roi_name, x1, x2, y1, y2 in rois:
                roi_gray = gray[y1:y2, x1:x2]
                roi_hsv = hsv[y1:y2, x1:x2]
                
                # Strategy 1: Edge detection within ROI
                edges = cv2.Canny(roi_gray, 50, 150)
                contours_edge, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours_edge:
                    area = cv2.contourArea(contour)
                    if 2000 < area < 100000:
                        _, _, cw, ch = cv2.boundingRect(contour)
                        if ch > cw * 0.8:  # Taller than wide
                            doorway_count += 1
                            break  # Only count once per ROI
                
                # Strategy 2: Bright areas (openings with light)
                lower_bright = np.array([0, 0, 190])
                upper_bright = np.array([180, 100, 255])
                mask_bright = cv2.inRange(roi_hsv, lower_bright, upper_bright)
                contours_bright, _ = cv2.findContours(mask_bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours_bright:
                    area = cv2.contourArea(contour)
                    if 3000 < area < 80000:
                        bright_count += 1
                        break  # Only count once per ROI
                
                # Strategy 3: Dark openings (looking into dark interior)
                lower_dark = np.array([0, 0, 10])
                upper_dark = np.array([180, 50, 40])
                mask_dark = cv2.inRange(roi_hsv, lower_dark, upper_dark)
                contours_dark, _ = cv2.findContours(mask_dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours_dark:
                    area = cv2.contourArea(contour)
                    if 4000 < area < 120000:
                        _, _, cw, ch = cv2.boundingRect(contour)
                        if 0.2 < (cw / ch) < 5.0:
                            dark_exits = min(dark_exits + 1, 2)
                            break  # Only count once per ROI
            
            # Combine detections
            archways = max(min(doorway_count, 1), min(bright_count, 1))
            open_doors = dark_exits
            closed_doors = 0
            
            # Return conservative counts
            total = min(archways + open_doors, 3)
            
            return {
                'closed_doors': closed_doors,
                'open_doors': open_doors,
                'archways': archways,
                'total': total
            }
        except Exception as e:
            return {'closed_doors': 0, 'open_doors': 0, 'archways': 0, 'total': 0}
    
    def _detect_door_state(self, screen):
        """
        Detect if there's a closed door (with 'open' interaction prompt) nearby.
        Looks specifically for the 'E [Open]' text that appears for doors and objects.
        Distinguishes between GREY (unavailable) and WHITE (ready to interact) text.
        
        Returns:
            dict with door state info:
            {
                'has_closable_door': bool - True if interaction prompt visible (any color),
                'has_open_prompt': bool - True if 'open' text is visible,
                'prompt_is_white': bool - True if prompt is WHITE (ready), False if GREY (waiting),
                'prompt_brightness': 'white'|'grey'|'none' - Color state of the prompt
            }
        """
        try:
            # The interaction prompt appears at center-bottom of screen
            # It's always: E [text] where text could be "Open", "Unlock", "Read", etc.
            # We look for a very specific pattern in a very specific location
            
            h, w = screen.shape[:2]
            # Interaction prompt typically centered horizontally, near bottom
            # In 1920x1080: approximately x=850-1070, y=1030-1070
            prompt_y_start = max(0, h - 80)   # Bottom 80 pixels
            prompt_y_end = min(h, h - 20)     # Stop 20 pixels from bottom
            prompt_x_start = max(0, w//2 - 200)   # Centered, ±200 pixels
            prompt_x_end = min(w, w//2 + 200)
            
            prompt_region = screen[prompt_y_start:prompt_y_end, prompt_x_start:prompt_x_end]
            
            if prompt_region.size == 0:
                return {
                    'has_closable_door': False, 
                    'has_open_prompt': False,
                    'prompt_is_white': False,
                    'prompt_brightness': 'none'
                }
            
            hsv = cv2.cvtColor(prompt_region, cv2.COLOR_RGB2HSV)
            
            # Look for WHITE text (ready to interact): high brightness (220+), low saturation
            lower_white = np.array([0, 0, 220])
            upper_white = np.array([180, 60, 255])
            mask_white = cv2.inRange(hsv, lower_white, upper_white)
            white_pixels = cv2.countNonZero(mask_white)
            
            # Look for GREY text (unavailable/waiting): medium-low brightness (100-180), very low saturation
            lower_grey = np.array([0, 0, 100])
            upper_grey = np.array([180, 30, 180])
            mask_grey = cv2.inRange(hsv, lower_grey, upper_grey)
            grey_pixels = cv2.countNonZero(mask_grey)
            
            # Threshold: need significant pixels to confirm text
            white_threshold = 250
            grey_threshold = 200
            
            has_white = white_pixels > white_threshold
            has_grey = grey_pixels > grey_threshold
            has_any = has_white or has_grey
            
            # Determine the state
            if has_white:
                brightness = 'white'
            elif has_grey:
                brightness = 'grey'
            else:
                brightness = 'none'
            
            return {
                'has_closable_door': has_any,
                'has_open_prompt': has_any,
                'prompt_is_white': has_white,
                'prompt_brightness': brightness
            }
        except Exception as e:
            return {
                'has_closable_door': False, 
                'has_open_prompt': False,
                'prompt_is_white': False,
                'prompt_brightness': 'none'
            }
    
    def _detect_quickslots(self, screen):
        """
        Detect which quickslots (0-7) have items
        Elden Ring quickslots are at the bottom of the screen in a horizontal row
        
        Returns:
            list of 8 bools [slot0_has_item, slot1_has_item, ...] 
            True = slot has an item, False = slot is empty
        """
        try:
            # Elden Ring quickslot UI is located at bottom-right
            # Approximate region: bottom row, right side
            # Screen is 1920x1080, quickslots are roughly at y=1000-1070, x=1550-1920
            h, w = screen.shape[:2]
            
            # Define quickslot region (bottom-right area)
            slot_y_start = max(0, h - 80)  # Bottom 80 pixels
            slot_y_end = h
            slot_x_start = max(0, w - 400)  # Right 400 pixels
            slot_x_end = w
            
            slot_region = screen[slot_y_start:slot_y_end, slot_x_start:slot_x_end]
            
            # Convert to HSV for item icon detection
            hsv = cv2.cvtColor(slot_region, cv2.COLOR_RGB2HSV)
            
            # Item icons typically have vibrant colors (not gray/neutral)
            # We detect non-empty slots by looking for colored pixels
            # Exclude pure gray/white/black (empty slots have no saturation or are white)
            
            # Create mask for "colored" pixels (high saturation = item icon)
            # Items have distinct colors, empty slots are neutral backgrounds
            lower_color = np.array([0, 30, 50])  # Min saturation for item icon
            upper_color = np.array([180, 255, 255])
            mask_colored = cv2.inRange(hsv, lower_color, upper_color)
            
            # Divide slot region into 8 equal parts (one per slot)
            slot_width = slot_region.shape[1] // 8
            quickslots = []
            
            for slot_idx in range(8):
                # Extract region for this slot
                slot_x_left = slot_idx * slot_width
                slot_x_right = (slot_idx + 1) * slot_width
                slot_mask = mask_colored[:, slot_x_left:slot_x_right]
                
                # Count colored pixels in this slot
                colored_pixel_count = cv2.countNonZero(slot_mask)
                
                # If slot has enough colored pixels, it has an item
                # Empty slots have very few colored pixels
                has_item = colored_pixel_count > 100  # Threshold for "has item"
                quickslots.append(has_item)
            
            return quickslots
            
        except Exception as e:
            # If detection fails, return all empty
            return [False] * 8
    
    def _detect_wizened_finger(self, screen):
        """
        Detect if Wizened Finger (key item) is in inventory
        Uses OCR to read item names from inventory screen for accurate detection
        
        Searches inventory screen for the text "Wizened Finger" or "Tarnished's Wizened Finger"
        Falls back to color detection if OCR fails
        
        Returns:
            bool: True if Wizened Finger is detected in inventory, False otherwise
        """
        try:
            # Check if pytesseract is available before importing
            try:
                import pytesseract
                from PIL import Image
                
                h, w = screen.shape[:2]
                
                # Convert BGR to RGB for PIL/pytesseract
                screen_rgb = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
                
                # OCR search in main inventory area (right 70% of screen, middle 60% vertically)
                # This is where item names are displayed
                inv_region_rgb = screen_rgb[int(h*0.15):int(h*0.85), int(w*0.3):w]
                inv_pil = Image.fromarray(inv_region_rgb.astype('uint8'))
                
                # Run OCR on inventory region
                try:
                    text = pytesseract.image_to_string(inv_pil)
                    
                    # Search for "Wizened Finger" in detected text (case-insensitive)
                    if "wizened finger" in text.lower():
                        return True
                    if "tarnished" in text.lower() and "finger" in text.lower():
                        return True
                except Exception as ocr_error:
                    # OCR failed, fall back to color detection
                    pass
            except ImportError:
                # pytesseract/PIL not installed - skip OCR, use color detection only
                pass
            
            # FALLBACK: Color-based detection if OCR unavailable
            # Look in quickslots for brownish-gold colored items
            slot_y_start = max(0, h - 80)
            slot_y_end = h
            slot_x_start = max(0, w - 400)
            slot_x_end = w
            
            slot_region = screen[slot_y_start:slot_y_end, slot_x_start:slot_x_end]
            hsv = cv2.cvtColor(slot_region, cv2.COLOR_RGB2HSV)
            
            # Wizened Finger is brownish-gold/tan colored
            lower_gold = np.array([10, 60, 80])
            upper_gold = np.array([30, 200, 220])
            mask_gold = cv2.inRange(hsv, lower_gold, upper_gold)
            
            # Check each quickslot for golden pixels
            slot_width = slot_region.shape[1] // 8
            for slot_idx in range(8):
                slot_x_left = slot_idx * slot_width
                slot_x_right = (slot_idx + 1) * slot_width
                slot_mask = mask_gold[:, slot_x_left:slot_x_right]
                
                golden_pixel_count = cv2.countNonZero(slot_mask)
                
                # If slot has enough golden pixels, it's likely the Wizened Finger
                if golden_pixel_count > 50:
                    return True
            
            # Color fallback: search main inventory area
            inv_region = screen[int(h*0.2):int(h*0.8), int(w*0.3):w]
            hsv_inv = cv2.cvtColor(inv_region, cv2.COLOR_RGB2HSV)
            mask_gold_inv = cv2.inRange(hsv_inv, lower_gold, upper_gold)
            
            golden_pixel_count_inv = cv2.countNonZero(mask_gold_inv)
            
            if golden_pixel_count_inv > 150:
                return True
            
            return False
            
        except Exception as e:
            # If detection fails completely, return False (safer than false positive)
            return False
    
    def _detect_outdoor_area(self, screen):
        """
        Detect if currently in an outdoor area (outside the Chapel)
        IMPROVED: Checks for both sky visibility AND outdoor vegetation/features
        More reliable distinction between chapel interior and outdoor area
        
        Returns:
            bool: True if likely outdoors, False if likely indoors
        """
        try:
            # Check the top portion of the screen for sky colors
            # Sky is typically bright blue (high value, low saturation) or bright gray
            top_region = screen[0:150, :]  # Top 150 pixels
            
            hsv = cv2.cvtColor(top_region, cv2.COLOR_RGB2HSV)
            
            # Bright areas (sky) have high Value component
            # Blue sky: 100-130 hue, medium saturation, high value
            # Gray sky: any hue, low saturation, high value (150+)
            
            # Detect bright pixels (sky)
            bright_mask = hsv[:, :, 2] > 150  # Value > 150
            bright_pixel_count = cv2.countNonZero(bright_mask)
            
            # If more than 20% of top region is bright, likely outdoors
            total_pixels = top_region.shape[0] * top_region.shape[1]
            brightness_ratio = bright_pixel_count / total_pixels
            
            # IMPROVED: Also check middle region for outdoor features
            # Outdoor areas tend to have more varied colors (grass, trees, vegetation)
            # Chapel interior is mostly uniform stone/dark colors
            middle_region = screen[150:400, :]
            middle_hsv = cv2.cvtColor(middle_region, cv2.COLOR_RGB2HSV)
            
            # Count color variety - outdoor areas have diverse greens/browns
            # Green (outdoor vegetation): hue 35-100
            green_mask = (middle_hsv[:, :, 0] > 35) & (middle_hsv[:, :, 0] < 100)
            green_count = cv2.countNonZero(green_mask)
            green_ratio = green_count / (middle_region.shape[0] * middle_region.shape[1])
            
            # Outdoor if: bright sky OR significant green vegetation visible
            is_outdoor = (brightness_ratio > 0.2) or (green_ratio > 0.1)
            
            return is_outdoor
            
        except Exception as e:
            return False
    
    def _detect_boss_health_bar(self, screen):
        """
        Detect if boss health bar is visible on screen.
        Boss health bar appears at bottom center of screen during boss fight.
        Typically red/dark red color with white borders.
        
        Returns:
            bool: True if boss health bar detected, False otherwise
        """
        try:
            # Check bottom portion of screen where boss health bar appears
            # Boss bar is typically at y: 1000-1080 (bottom 80 pixels of 1080p screen)
            bottom_region = screen[-100:, :]  # Bottom 100 pixels
            
            # Boss health bar is red/dark color
            hsv = cv2.cvtColor(bottom_region, cv2.COLOR_RGB2HSV)
            
            # Red colors in HSV: 0-10 or 170-180 hue, high saturation
            lower_red1 = np.array([0, 100, 100])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 100, 100])
            upper_red2 = np.array([180, 255, 255])
            
            mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
            
            # Check if there's a significant red area (boss bar)
            red_pixels = cv2.countNonZero(mask_red)
            total_pixels = bottom_region.shape[0] * bottom_region.shape[1]
            red_ratio = red_pixels / total_pixels
            
            # If more than 5% of bottom region is red, boss bar is visible
            return red_ratio > 0.05
            
        except Exception as e:
            return False
    
    def _detect_fog_wall(self, screen):
        """
        Detect if golden fog wall (boss arena entrance) is visible.
        Fog walls are golden/yellow color and appear as vertical barriers.
        
        Returns:
            bool: True if fog wall detected, False otherwise
        """
        try:
            # Golden fog has yellow/gold color
            hsv = cv2.cvtColor(screen, cv2.COLOR_RGB2HSV)
            
            # Gold/yellow: hue 20-40, high saturation, medium-high value
            lower_gold = np.array([20, 100, 100])
            upper_gold = np.array([40, 255, 200])
            
            mask_gold = cv2.inRange(hsv, lower_gold, upper_gold)
            
            # Find contours of golden areas
            contours, _ = cv2.findContours(mask_gold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Fog walls are large vertical structures
            for contour in contours:
                area = cv2.contourArea(contour)
                # Fog wall area should be significant (> 10000 pixels)
                if area > 10000:
                    x, y, w, h = cv2.boundingRect(contour)
                    # Fog wall is typically tall and narrow (h > w)
                    if h > w * 1.5:
                        return True
            
            return False
            
        except Exception as e:
            return False
    
    def _detect_roof(self, screen):
        """
        Detect if there's a roof/ceiling above the character (indicates indoors).
        
        Roofs typically have:
        - Horizontal/parallel line structure (edges)
        - Uniform texture (low variation in color within the roof area)
        - Located in upper portion of screen
        - Similar color to building/stonework (not bright sky)
        
        Returns:
            bool: True if roof/ceiling detected, False otherwise
        """
        try:
            # Look at top 1/3 of screen where ceiling would be visible
            roof_region = screen[0:screen.shape[0]//3, :]
            
            # Convert to grayscale for edge detection
            gray = cv2.cvtColor(roof_region, cv2.COLOR_RGB2GRAY)
            
            # Detect edges (horizontal lines of a ceiling)
            edges = cv2.Canny(gray, 50, 150)
            
            # Look for horizontal edges (ceiling/roof edges are typically horizontal)
            # Use morphological operations to emphasize horizontal structures
            kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 3))
            horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel_horizontal)
            
            # Count horizontal line pixels
            horizontal_pixels = cv2.countNonZero(horizontal_lines)
            total_pixels = roof_region.shape[0] * roof_region.shape[1]
            horizontal_ratio = horizontal_pixels / total_pixels
            
            # Also check for uniformity (ceiling is typically uniform color/texture)
            # Calculate color variance in roof region
            hsv = cv2.cvtColor(roof_region, cv2.COLOR_RGB2HSV)
            color_std = np.std(hsv[:, :, 2])  # Std dev of Value channel
            
            # Roof detected if:
            # - Horizontal lines present (edges) AND
            # - Color is relatively uniform (low std dev - not bright sky) AND
            # - Not too bright (not open sky)
            has_horizontal_structure = horizontal_ratio > 0.01
            is_uniform_color = color_std < 40  # Lower std = more uniform
            is_not_bright_sky = np.mean(gray) < 200  # Not very bright
            
            roof_detected = has_horizontal_structure and is_uniform_color and is_not_bright_sky
            
            return roof_detected
            
        except Exception as e:
            return False
    
    def _detect_status_buildup(self, screen):
        """
        Detect if a status effect buildup bar is visible (poison, rot, bleed, scarlet rot, etc.)
        Status buildup bars appear as colored fill indicators on screen, typically in bottom-left area
        
        Returns:
            bool: True if status buildup bar detected, False otherwise
        """
        try:
            # Status bars can be various colors: purple (poison), green (rot), red (bleed), etc.
            # For now, look for any highly saturated colored bar in status area (below health/stamina bars)
            # More specific detection can be added once we see actual status bars in gameplay
            
            # Status bar appears below stamina bar, typically in bottom-left area
            # Search region: bottom-left corner (similar to health/stamina)
            h, w = screen.shape[:2]
            status_roi = screen[int(h * 0.80):h, 0:int(w * 0.25)]
            
            if status_roi.size == 0:
                return False
            
            # Convert to HSV for color detection
            hsv = cv2.cvtColor(status_roi, cv2.COLOR_RGB2HSV)
            
            # Look for high saturation areas (status effects are typically vibrant colors)
            # This is a broad detector - will catch any highly saturated color bar
            # Can be refined once we see actual status buildup bars
            saturation = hsv[:, :, 1]
            
            # If significant area has high saturation (status bar typically fills with color)
            high_sat_pixels = np.sum(saturation > 150)
            total_pixels = saturation.size
            sat_ratio = high_sat_pixels / total_pixels
            
            # If more than 2% of status region is highly saturated, status bar likely visible
            return sat_ratio > 0.02
            
        except Exception as e:
            return False
    
    def _detect_ground_items(self, screen):
        """
        Detect if there are glowing items visible anywhere on screen.
        Items can be on ground, ledges, corpses, in chests, etc.
        Items typically have a golden/bright glow that stands out from environment.
        
        Returns:
            bool: True if items detected, False otherwise
        """
        try:
            # Items have bright golden/white glow regardless of location
            # Look for bright pixels across most of the screen
            
            hsv = cv2.cvtColor(screen, cv2.COLOR_RGB2HSV)
            
            # Extract value channel (brightness)
            value = hsv[:, :, 2]
            
            # Items typically have high brightness (>200 in 0-255 scale)
            # Search across most of the screen, excluding UI elements at bottom and top
            h, w = screen.shape[:2]
            
            # Search region: full width, middle 75% of height (avoid HUD at bottom, sky at very top)
            # Bottom 15% excluded for UI, top 10% excluded for sky/distant objects
            item_roi = value[int(h * 0.10):int(h * 0.85), :]
            
            if item_roi.size == 0:
                return False
            
            # Count very bright pixels (item glow)
            # Items have distinctive bright appearance
            bright_pixels = np.sum(item_roi > 200)
            total_pixels = item_roi.size
            brightness_ratio = bright_pixels / total_pixels
            
            # If more than 0.3% of visible area is very bright, items likely present
            # Lowered threshold to catch items at various distances/locations
            return brightness_ratio > 0.003
            
        except Exception as e:
            return False
    
    def _detect_in_combat(self, screen, health_dropped=False, status_buildup_visible=False):
        """
        Detect if AI is currently in combat.
        Combat is detected by:
        - Boss health bar visible (red bar at bottom)
        - Fog wall visible (boss arena)
        - Health dropped recently (damage from enemies) AND no status buildup (environmental damage)
        
        Args:
            screen: screenshot
            health_dropped: bool - True if health decreased this frame
            status_buildup_visible: bool - True if status buildup bar is visible
        
        Returns:
            bool: True if in combat, False otherwise
        """
        try:
            # Combat if boss health is visible
            boss_health = self._detect_boss_health_bar(screen)
            if boss_health:
                return True
            
            # Combat if fog wall visible (in boss arena)
            fog_wall = self._detect_fog_wall(screen)
            if fog_wall:
                return True
            
            # Combat if taking damage that's NOT from status effects
            # (status buildup visible = poison/rot/bleed damage = environmental)
            if health_dropped and not status_buildup_visible:
                # Health dropped without status effect = enemy damage
                return True
            
            # If recently took non-environmental damage, stay in combat for a few frames
            if self.recent_damage_frames < 30:  # ~0.5 seconds at 60fps
                return True
            
            return False
            
        except Exception as e:
            return False

    
    # Input actions
    def move_character(self, x, y):
        """
        Move character (WASD controls)
        
        Args:
            x, y: normalized direction (-1 to 1)
        """
        keys_to_press = []
        
        # Forward/backward
        if y > 0.3:
            keys_to_press.append(self.CONTROLS['forward'])  # W
        elif y < -0.3:
            keys_to_press.append(self.CONTROLS['backward'])  # S
        
        # Left/right
        if x > 0.3:
            keys_to_press.append(self.CONTROLS['right'])  # D
        elif x < -0.3:
            keys_to_press.append(self.CONTROLS['left'])  # A
        
        # Press all movement keys
        for key in keys_to_press:
            self.keyboard.press(key)
        
        # HOLD keys for 0.5 seconds to ensure movement happens
        # (Elden Ring needs sustained key press, not a tap)
        time.sleep(0.5)
        
        # Release all keys
        for key in keys_to_press:
            self.keyboard.release(key)
    
    def backstep(self):
        """
        Perform backstep/dodge/dash
        Default: Spacebar
        """
        self.keyboard.press(self.CONTROLS['backstep'])
        time.sleep(0.02)
        self.keyboard.release(self.CONTROLS['backstep'])
    
    def attack(self):
        """
        Perform normal attack
        Default: Left Mouse Button
        """
        self.mouse.click(self.CONTROLS['normal_attack'], 1)
        time.sleep(0.1)
    
    def heavy_attack(self):
        """
        Perform heavy attack
        Default: Shift + Left Click
        """
        self.keyboard.press(self.CONTROLS['heavy_attack'])  # Shift
        time.sleep(0.02)
        self.mouse.click(self.CONTROLS['normal_attack'], 1)
        time.sleep(0.05)
        self.keyboard.release(self.CONTROLS['heavy_attack'])
    
    def guard(self):
        """
        Guard/Block with shield
        Default: Right Mouse Button
        """
        self.mouse.click(self.CONTROLS['guard'], 1)
        time.sleep(0.1)
    
    def skill(self):
        """
        Use weapon skill/ability
        Default: Shift + Right Click
        """
        self.keyboard.press(self.CONTROLS['skill'])  # Shift
        time.sleep(0.02)
        self.mouse.click(self.CONTROLS['guard'], 1)  # Right Click
        time.sleep(0.05)
        self.keyboard.release(self.CONTROLS['skill'])
    
    def use_item(self):
        """
        Use equipped item
        Default: R key
        """
        self.keyboard.press(self.CONTROLS['use_item'])
        time.sleep(0.05)
        self.keyboard.release(self.CONTROLS['use_item'])
    
    def two_hand_weapon(self):
        """
        Two-hand equipped weapon
        Default: Hold E + Left Click
        """
        self.keyboard.press(self.CONTROLS['two_hand'])  # E
        time.sleep(0.02)
        self.mouse.click(self.CONTROLS['normal_attack'], 1)
        time.sleep(0.05)
        self.keyboard.release(self.CONTROLS['two_hand'])
    
    def lock_on(self):
        """
        Toggle lock-on / Reset camera
        Default: Q or Middle Mouse Button
        """
        self.keyboard.press(self.CONTROLS['lock_on'])
        time.sleep(0.05)
        self.keyboard.release(self.CONTROLS['lock_on'])
    
    def reset_camera(self):
        """
        Reset camera position
        Default: Middle Mouse Button
        """
        self.mouse.click(self.CONTROLS['reset_camera'], 1)
        time.sleep(0.05)
    
    def jump(self):
        """
        Jump
        Default: F key
        """
        self.keyboard.press(self.CONTROLS['jump'])
        time.sleep(0.02)
        self.keyboard.release(self.CONTROLS['jump'])
    
    def crouch(self):
        """
        Crouch/Stand toggle
        Default: X key
        """
        self.keyboard.press(self.CONTROLS['crouch'])
        time.sleep(0.05)
        self.keyboard.release(self.CONTROLS['crouch'])
    
    def interact(self):
        """
        Interact with NPCs, objects, doors, items, or perform event action
        Default: E key
        """
        self.keyboard.press(self.CONTROLS['event_action'])
        time.sleep(0.05)
        self.keyboard.release(self.CONTROLS['event_action'])
    
    def summon_mount(self):
        """
        Summon the mount (Torrent in Elden Ring)
        Uses R key (Use Item) to summon Spectral Steed Whistle
        """
        self.keyboard.press(self.CONTROLS['use_item'])
        time.sleep(0.05)
        self.keyboard.release(self.CONTROLS['use_item'])
    
    # Menu and navigation controls
    def open_map(self):
        """
        Open map
        Default: G key
        """
        self.keyboard.press(self.CONTROLS['map'])
        time.sleep(0.05)
        self.keyboard.release(self.CONTROLS['map'])
    
    def open_main_menu(self):
        """
        Open main menu
        Default: Esc key
        """
        self.keyboard.press(self.CONTROLS['main_menu'])
        time.sleep(0.05)
        self.keyboard.release(self.CONTROLS['main_menu'])
    
    def switch_spell(self):
        """
        Switch Sorcery/Incantation
        Default: Up Arrow
        """
        self.keyboard.press(self.CONTROLS['switch_sorcery'])
        time.sleep(0.05)
        self.keyboard.release(self.CONTROLS['switch_sorcery'])
    
    def switch_item(self):
        """
        Switch Item
        Default: Down Arrow
        """
        self.keyboard.press(self.CONTROLS['switch_item'])
        time.sleep(0.05)
        self.keyboard.release(self.CONTROLS['switch_item'])
    
    def switch_right_hand_weapon(self):
        """
        Switch Right-Hand Armament
        Default: Right Arrow or Shift + Scroll Up
        """
        self.keyboard.press(self.CONTROLS['switch_right_hand'])
        time.sleep(0.05)
        self.keyboard.release(self.CONTROLS['switch_right_hand'])
    
    def switch_left_hand_weapon(self):
        """
        Switch Left-Hand Armament
        Default: Left Arrow or Shift + Scroll Down
        """
        self.keyboard.press(self.CONTROLS['switch_left_hand'])
        time.sleep(0.05)
        self.keyboard.release(self.CONTROLS['switch_left_hand'])
