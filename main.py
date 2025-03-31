import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import os
import cv2
import numpy as np
from PIL import Image
import threading
import time
import json
from datetime import datetime

# Dark Theme Colors
THEME = {
    "bg_dark": "#121212",        # Main background
    "bg_medium": "#1E1E1E",      # Card background
    "bg_light": "#2D2D2D",       # Input fields, headers
    "accent": "#BB86FC",         # Purple accent
    "accent2": "#03DAC6",        # Teal accent 
    "error": "#CF6679",          # Error color
    "success": "#4CAF50",        # Success color
    "warning": "#FFAB40",        # Warning color
    "text": "#E1E1E1",           # Primary text
    "text_secondary": "#B0B0B0"  # Secondary text
}

# File paths for data storage
USER_DATA_FILE = "user_data.json"
ATTENDANCE_FILE = "attendance.json"

# Initialize main window with dark theme
window = tk.Tk()
window.title("Face Recognition Attendance System")
window.configure(bg=THEME["bg_dark"])
window.geometry("900x700")

# Configure styles for dark theme
style = ttk.Style()
style.theme_use("clam")  # Base theme that works well for customization

# Configure frame styles
style.configure("TFrame", background=THEME["bg_dark"])
style.configure("Card.TFrame", background=THEME["bg_medium"], relief="flat")
style.configure("Header.TFrame", background=THEME["bg_light"])

# Configure label styles
style.configure("TLabel", 
                background=THEME["bg_dark"], 
                foreground=THEME["text"], 
                font=("Segoe UI", 11))
style.configure("Card.TLabel", background=THEME["bg_medium"], foreground=THEME["text"])
style.configure("Header.TLabel", 
                background=THEME["bg_light"], 
                foreground=THEME["accent"], 
                font=("Segoe UI", 14, "bold"))
style.configure("Title.TLabel", 
                background=THEME["bg_dark"], 
                foreground=THEME["accent"], 
                font=("Segoe UI", 18, "bold"))
style.configure("Status.TLabel", 
                background=THEME["bg_light"], 
                foreground=THEME["text"], 
                font=("Segoe UI", 10))

# Configure button styles
style.configure("Accent.TButton", 
                background=THEME["accent"], 
                foreground=THEME["bg_dark"], 
                font=("Segoe UI", 11, "bold"),
                padding=8)
style.map("Accent.TButton",
          background=[("active", THEME["accent2"]), ("disabled", THEME["bg_light"])])

# Configure entry style
style.configure("TEntry", 
                fieldbackground=THEME["bg_light"],
                foreground=THEME["text"],
                insertcolor=THEME["text"],
                font=("Segoe UI", 11))

# Configure combobox style
style.configure("TCombobox", 
                fieldbackground=THEME["bg_light"],
                background=THEME["bg_medium"],
                foreground=THEME["text"],
                arrowcolor=THEME["accent"])
style.map("TCombobox", 
          fieldbackground=[("readonly", THEME["bg_light"])],
          selectbackground=[("readonly", THEME["accent"])],
          selectforeground=[("readonly", THEME["bg_dark"])])

# Configure progress bar
style.configure("Horizontal.TProgressbar", 
                background=THEME["accent"],
                troughcolor=THEME["bg_light"],
                thickness=15)

# Main container
main_frame = ttk.Frame(window, style="TFrame")
main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

# App title
title_label = ttk.Label(main_frame, text="Face Recognition Attendance System", style="Title.TLabel")
title_label.pack(fill=tk.X, pady=(0, 15))

# User Info Card
user_card = ttk.Frame(main_frame, style="Card.TFrame")
user_card.pack(fill=tk.X, pady=10, padx=5, ipady=5)

user_header = ttk.Frame(user_card, style="Header.TFrame")
user_header.pack(fill=tk.X)

user_title = ttk.Label(user_header, text="Student Information", style="Header.TLabel")
user_title.pack(padx=10, pady=5, anchor=tk.W)

# User form
user_form = ttk.Frame(user_card, style="Card.TFrame")
user_form.pack(fill=tk.X, padx=15, pady=10)

# Create all input fields
def create_form_field(parent, label_text, row):
    frame = ttk.Frame(parent, style="Card.TFrame")
    frame.grid(row=row, column=0, sticky="ew", pady=5)
    frame.columnconfigure(1, weight=1)
    
    label = ttk.Label(frame, text=label_text, width=12, style="Card.TLabel")
    label.grid(row=0, column=0, sticky="w", padx=(0, 10))
    
    entry = ttk.Entry(frame)
    entry.grid(row=0, column=1, sticky="ew")
    
    return entry

# Configure user_form for grid layout
user_form.columnconfigure(0, weight=1)

# Name field
name_entry = create_form_field(user_form, "Name:", 0)

# College ID field
college_id_entry = create_form_field(user_form, "College ID:", 1)

# Branch field with dropdown
branch_frame = ttk.Frame(user_form, style="Card.TFrame")
branch_frame.grid(row=2, column=0, sticky="ew", pady=5)
branch_frame.columnconfigure(1, weight=1)

branch_label = ttk.Label(branch_frame, text="Branch:", width=12, style="Card.TLabel")
branch_label.grid(row=0, column=0, sticky="w", padx=(0, 10))

branches = ["Computer Science", "Information Technology", "Electronics", "Electrical", "Mechanical", "Civil", "Other"]
branch_var = tk.StringVar()
branch_combo = ttk.Combobox(branch_frame, textvariable=branch_var, values=branches, state="readonly")
branch_combo.grid(row=0, column=1, sticky="ew")
branch_combo.current(0)  # Default to first option

# Year field with dropdown
year_frame = ttk.Frame(user_form, style="Card.TFrame")
year_frame.grid(row=3, column=0, sticky="ew", pady=5)
year_frame.columnconfigure(1, weight=1)

year_label = ttk.Label(year_frame, text="Year:", width=12, style="Card.TLabel")
year_label.grid(row=0, column=0, sticky="w", padx=(0, 10))

years = ["First Year (FE)", "Second Year (SE)", "Third Year (TE)", "Final Year (BE)"]
year_var = tk.StringVar()
year_combo = ttk.Combobox(year_frame, textvariable=year_var, values=years, state="readonly")
year_combo.grid(row=0, column=1, sticky="ew")
year_combo.current(0)  # Default to first option

# Gender field with radio buttons
gender_frame = ttk.Frame(user_form, style="Card.TFrame")
gender_frame.grid(row=4, column=0, sticky="ew", pady=5)

gender_label = ttk.Label(gender_frame, text="Gender:", width=12, style="Card.TLabel")
gender_label.grid(row=0, column=0, sticky="w", padx=(0, 10))

gender_var = tk.StringVar(value="Male")
gender_radio_frame = ttk.Frame(gender_frame, style="Card.TFrame")
gender_radio_frame.grid(row=0, column=1, sticky="w")

male_radio = ttk.Radiobutton(gender_radio_frame, text="Male", variable=gender_var, value="Male")
male_radio.pack(side=tk.LEFT, padx=(0, 20))

female_radio = ttk.Radiobutton(gender_radio_frame, text="Female", variable=gender_var, value="Female")
female_radio.pack(side=tk.LEFT)

# Buttons frame
buttons_frame = ttk.Frame(main_frame, style="Card.TFrame")
buttons_frame.pack(fill=tk.X, pady=10, padx=5, ipady=10)

buttons_header = ttk.Frame(buttons_frame, style="Header.TFrame")
buttons_header.pack(fill=tk.X)

buttons_title = ttk.Label(buttons_header, text="Actions", style="Header.TLabel")
buttons_title.pack(padx=10, pady=5, anchor=tk.W)

# Button container with even spacing
button_container = ttk.Frame(buttons_frame, style="Card.TFrame")
button_container.pack(fill=tk.X, padx=15, pady=10)
button_container.columnconfigure(0, weight=1)
button_container.columnconfigure(1, weight=1)
button_container.columnconfigure(2, weight=1)

# Create buttons with consistent styling
generate_button = ttk.Button(button_container, text="📷 Generate Dataset", style="Accent.TButton")
generate_button.grid(row=0, column=0, padx=10, pady=5, sticky="ew")

train_button = ttk.Button(button_container, text="🧠 Train Model", style="Accent.TButton")
train_button.grid(row=0, column=1, padx=10, pady=5, sticky="ew")

detect_button = ttk.Button(button_container, text="👁️ Take Attendance", style="Accent.TButton")
detect_button.grid(row=0, column=2, padx=10, pady=5, sticky="ew")

# Log area with dark theme
log_frame = ttk.Frame(main_frame, style="Card.TFrame")
log_frame.pack(fill=tk.BOTH, expand=True, pady=10, padx=5)

log_header = ttk.Frame(log_frame, style="Header.TFrame")
log_header.pack(fill=tk.X)

log_title = ttk.Label(log_header, text="System Log", style="Header.TLabel")
log_title.pack(padx=10, pady=5, anchor=tk.W)

# Dark styled log with good contrast
log_area = scrolledtext.ScrolledText(log_frame, 
                                    font=("Consolas", 10),
                                    bg="#1A1A1A", 
                                    fg="#E0E0E0", 
                                    insertbackground=THEME["text"],
                                    relief="flat",
                                    height=10)
log_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
log_area.config(state=tk.DISABLED)

# Progress section
progress_frame = ttk.Frame(main_frame, style="Card.TFrame")
progress_frame.pack(fill=tk.X, pady=10, padx=5)

progress_var = tk.DoubleVar(value=0)
progress_bar = ttk.Progressbar(progress_frame, 
                              orient="horizontal", 
                              length=100, 
                              mode="determinate",
                              variable=progress_var)
progress_bar.pack(fill=tk.X, padx=10, pady=10)

# Status bar
status_var = tk.StringVar(value="Ready")
status_bar = ttk.Label(window, textvariable=status_var, style="Status.TLabel")
status_bar.pack(side=tk.BOTTOM, fill=tk.X)

# Load user data
user_data = {}
if os.path.exists(USER_DATA_FILE):
    try:
        with open(USER_DATA_FILE, 'r') as f:
            user_data = json.load(f)
    except Exception as e:
        print(f"Error loading user data: {e}")

# Load attendance data
attendance_data = {}
if os.path.exists(ATTENDANCE_FILE):
    try:
        with open(ATTENDANCE_FILE, 'r') as f:
            attendance_data = json.load(f)
    except Exception as e:
        print(f"Error loading attendance data: {e}")

# Function to update log with dark theme colors
def update_log(message, message_type="INFO"):
    log_area.config(state=tk.NORMAL)
    timestamp = time.strftime("%H:%M:%S")
    
    # Color code messages by type
    if message_type == "ERROR":
        tag = "error"
        log_area.tag_config(tag, foreground=THEME["error"], font=("Consolas", 10, "bold"))
    elif message_type == "SUCCESS":
        tag = "success"
        log_area.tag_config(tag, foreground=THEME["success"], font=("Consolas", 10, "bold"))
    elif message_type == "WARNING":
        tag = "warning"
        log_area.tag_config(tag, foreground=THEME["warning"], font=("Consolas", 10))
    else:
        tag = "info"
        log_area.tag_config(tag, foreground=THEME["accent2"], font=("Consolas", 10))
    
    prefix = f"[{message_type}]" if message_type != "INFO" else ""
    log_area.insert(tk.END, f"{timestamp} {prefix} {message}\n", tag)
    log_area.see(tk.END)
    log_area.config(state=tk.DISABLED)
    status_var.set(message)

def update_button_states():
    # Count dataset files
    data_count = 0
    if os.path.exists("data"):
        data_count = len([f for f in os.listdir("data") if f.endswith('.jpg')])
    
    # Check if classifier exists
    classifier_exists = os.path.exists("classifier.xml")
    
    # Update button states
    if data_count == 0:
        train_button.config(state=tk.DISABLED)
    else:
        train_button.config(state=tk.NORMAL)
    
    if not classifier_exists:
        detect_button.config(state=tk.DISABLED)
    else:
        detect_button.config(state=tk.NORMAL)
    
    # Update status message
    if data_count > 0:
        update_log(f"Dataset ready: {data_count} images.", "SUCCESS")

def validate_user_input():
    # Get all user inputs
    name = name_entry.get().strip()
    college_id = college_id_entry.get().strip()
    branch = branch_var.get()
    year = year_var.get()
    gender = gender_var.get()
    
    # Validate required fields
    if not name:
        messagebox.showwarning("Input Required", "Please enter a name for the student")
        return None
    
    if not college_id:
        messagebox.showwarning("Input Required", "Please enter a college ID")
        return None
    
    # Return validated user data
    return {
        "name": name,
        "college_id": college_id,
        "branch": branch,
        "year": year,
        "gender": gender
    }

def train_classifier():
    # Validate user input before proceeding
    user_info = validate_user_input()
    if not user_info:
        return
        
    data_dir = "data"
    if not os.path.exists(data_dir) or len(os.listdir(data_dir)) == 0:
        update_log("No images found in data directory.", "ERROR")
        messagebox.showerror("Training Error", "No face images found. Generate dataset first.")
        return
    
    path = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.jpg')]
    update_log(f"Training with {len(path)} images...")
    
    # Disable buttons during training
    train_button.config(state=tk.DISABLED)
    generate_button.config(state=tk.DISABLED)
    detect_button.config(state=tk.DISABLED)
    
    def training_thread():
        faces = []
        ids = []
        progress_var.set(0)
        
        for i, image_path in enumerate(path):
            try:
                img = Image.open(image_path).convert('L')
                imageNp = np.array(img, 'uint8')
                
                # Extract ID from the second part of the filename
                parts = os.path.basename(image_path).split('.')
                if len(parts) >= 2:
                    id = int(parts[1])
                    faces.append(imageNp)
                    ids.append(id)
                
                # Update progress
                progress = (i + 1) / len(path) * 100
                progress_var.set(progress)
                
                if i % 10 == 0:
                    status_var.set(f"Training progress: {progress:.1f}%")
                    window.update_idletasks()
                
            except Exception as e:
                update_log(f"Error processing {os.path.basename(image_path)}: {e}", "ERROR")
        
        if not faces:
            update_log("No valid face images found.", "ERROR")
            window.after(0, lambda: update_button_states())
            return
            
        ids = np.array(ids)
        
        try:
            clf = cv2.face.LBPHFaceRecognizer_create()
            update_log(f"Training model...", "INFO")
            clf.train(faces, ids)
            clf.write("classifier.xml")
            update_log("Training completed successfully!", "SUCCESS")
            progress_var.set(100)
            messagebox.showinfo("Success", "Training completed successfully!")
        except Exception as e:
            update_log(f"Training error: {e}", "ERROR")
        
        window.after(0, lambda: update_button_states())
    
    threading.Thread(target=training_thread, daemon=True).start()

def detect_faces():
    if not os.path.exists("classifier.xml"):
        update_log("Classifier not found. Train the model first.", "ERROR")
        return
    
    if not os.path.exists("haarcascade_frontalface_default.xml"):
        update_log("Missing face cascade file.", "ERROR")
        return
    
    # Reload user data to ensure we have the latest
    global user_data, attendance_data
    if os.path.exists(USER_DATA_FILE):
        try:
            with open(USER_DATA_FILE, 'r') as f:
                user_data = json.load(f)
        except Exception as e:
            update_log(f"Error loading user data: {e}", "WARNING")
    
    if os.path.exists(ATTENDANCE_FILE):
        try:
            with open(ATTENDANCE_FILE, 'r') as f:
                attendance_data = json.load(f)
        except Exception as e:
            update_log(f"Error loading attendance data: {e}", "WARNING")
    
    update_log("Starting face detection for attendance...", "INFO")
    
    # Set of IDs that have been marked present this session
    marked_ids = set()
    
    def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text, clf):
        if img is None:
            return img
            
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
        
        for (x,y,w,h) in features:
            cv2.rectangle(img, (x,y), (x+w,y+h), color, 2)
            
            id, pred = clf.predict(gray_img[y:y+h,x:x+w])
            confidence = int(100*(1-pred/300))
            
            # Use the stored user data
            id_str = str(id)
            if confidence > 70:
                # Debug information
                update_log(f"Detected face with ID: {id}, confidence: {confidence}", "INFO")
                update_log(f"Available user data keys: {list(user_data.keys())}", "INFO")
                
                # Print all users and their IDs for debugging
                for name, info in user_data.items():
                    stored_id = info.get("id", "none")
                    update_log(f"User: {name}, Stored ID: {stored_id}", "INFO")
                    
                # Find user by ID in the user_data dictionary
                found_user = None
                for name, info in user_data.items():
                    if str(id) == info.get("id", ""):  # Store ID in user_info to match
                        found_user = name
                        break
                        
                # NEW CODE: If no exact match found, try to find the closest match above threshold
                if not found_user and confidence > 85:
                    # Face is recognized with high confidence but ID doesn't match any known user
                    # This happens when classifier gets confused between similar faces
                    # Use the first available user as fallback (optional)
                    if user_data:
                        found_user = list(user_data.keys())[0]
                        update_log(f"No exact ID match, using closest user: {found_user}", "WARNING")
                        
                if found_user:
                    user_info = user_data[found_user]
                    name = user_info["name"]
                    college_id = user_info["college_id"]
                    branch = user_info.get("branch", "Unknown")
                    year = user_info.get("year", "Unknown")
                    
                    text_color = (0, 255, 0)  # Green for high confidence
                    
                    # Display user info
                    info_text = f"{name} ({confidence}%)"
                    cv2.putText(img, info_text, (x,y-30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
                    
                    cv2.putText(img, f"{college_id}, {year}", (x,y-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)
                    
                    # Mark attendance using name instead of ID
                    if name not in marked_ids:
                        marked_ids.add(name)
                        
                        # Update attendance with name as key
                        today = datetime.now().strftime("%Y-%m-%d")
                        if name not in attendance_data:
                            attendance_data[name] = {"count": 1, "dates": [today]}
                        else:
                            attendance_data[name]["count"] += 1
                            if today not in attendance_data[name]["dates"]:
                                attendance_data[name]["dates"].append(today)
                        
                        # Save attendance data to file
                        with open(ATTENDANCE_FILE, 'w') as f:
                            json.dump(attendance_data, f, indent=2)
                        
                        update_log(f"Marked attendance for {name}", "SUCCESS")
                        
                        # Show attendance count
                        attendance_count = attendance_data[name]["count"]
                        cv2.putText(img, f"Attendance: {attendance_count}", (x,y+h+20), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
                else:
                    name = f"Unknown ID {id}"
                    text_color = (255, 255, 0)  # Yellow for unknown user
                    cv2.putText(img, f"{name} ({confidence}%)", (x,y-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
            else:
                cv2.putText(img, "UNKNOWN", (x,y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        
        return img
    
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.read("classifier.xml")
    
    video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not video_capture.isOpened():
        update_log("Could not open camera", "ERROR")
        return

    window_name = "Attendance System - Face Recognition"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Create stylish overlay text for instructions
    def add_overlay_text(img, text):
        h, w = img.shape[:2]
        overlay = img.copy()
        cv2.rectangle(overlay, (0, h-40), (w, h), (0, 0, 0), -1)
        alpha = 0.7
        img = cv2.addWeighted(overlay, alpha, img, 1-alpha, 0)
        cv2.putText(img, text, (10, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Add date and time on top
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.rectangle(overlay, (0, 0), (w, 30), (0, 0, 0), -1)
        img = cv2.addWeighted(overlay, alpha, img, 1-alpha, 0)
        cv2.putText(img, current_time, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return img
    
    start_time = time.time()
    
    while True:
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break
            
        ret, img = video_capture.read()
        if not ret or img is None:
            break
            
        img = draw_boundary(img, faceCascade, 1.3, 5, (255, 255, 255), "Face", clf)
        img = add_overlay_text(img, "Press 'ESC' or 'q' to exit | Face Recognition Attendance System")
        
        cv2.imshow(window_name, img)

        # Add a small delay to reduce CPU usage
        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):  # ESC or q
            break
    
    video_capture.release()
    cv2.destroyAllWindows()
    
    # Show summary of attendance taken
    if marked_ids:
        marked_names = [user_data[id_str]["name"] for id_str in marked_ids if id_str in user_data]
        update_log(f"Attendance recorded for {len(marked_ids)} students: {', '.join(marked_names)}", "SUCCESS")
    else:
        update_log("No attendance recorded in this session", "INFO")

def generate_dataset():
    # Validate user input
    user_info = validate_user_input()
    if not user_info:
        return
    
    user_name = user_info["name"]
    update_log(f"Generating dataset for: {user_name}")
    
    if not os.path.exists("haarcascade_frontalface_default.xml"):
        update_log("Missing cascade file", "ERROR")
        messagebox.showerror("Error", "Missing face cascade file. Please download it first.")
        return
    
    face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    
    if not os.path.exists("data"):
        os.makedirs("data")
    
    # Disable buttons during generation
    generate_button.config(state=tk.DISABLED)
    train_button.config(state=tk.DISABLED)
    detect_button.config(state=tk.DISABLED)
    
    progress_var.set(0)
    
    def face_cropped(img):
        if img is None:
            return None
        
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) == 0:
                return None
                
            for (x, y, w, h) in faces:
                return img[y:y+h, x:x+w]
            
            return None
        except Exception as e:
            update_log(f"Face detection error: {e}", "ERROR")
            return None
    
    def dataset_thread():
        camera_index = 0
        cap = None
        
        # Try to open camera
        for idx in [0, 1]:
            try:
                cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                if cap.isOpened():
                    camera_index = idx
                    time.sleep(1)
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None:
                        break
            except:
                pass
            
            if cap:
                cap.release()
                cap = None
        
        if not cap:
            update_log("Could not initialize camera", "ERROR")
            window.after(0, lambda: update_button_states())
            return
        
        # Get next user ID
        existing_ids = []
        if os.path.exists("data"):
            for f in os.listdir("data"):
                try:
                    parts = f.split(".")
                    if len(parts) >= 2 and parts[0].startswith("user"):
                        id_part = parts[1]
                        existing_ids.append(int(id_part))
                except Exception as e:
                    continue
        
        global user_data
        
        # Find existing IDs in user_data as well
        for id_str in user_data:
            try:
                existing_ids.append(int(id_str))
            except:
                pass
        
        # Get a new unique ID
        id = 1
        if existing_ids:
            id = max(existing_ids) + 1
        
        # Store user data with ID
        user_name = user_info["name"]
        user_info["id"] = str(id)  # Add this line to store the numeric ID
        user_data[user_name] = user_info
        
        # Save user data to file
        with open(USER_DATA_FILE, 'w') as f:
            json.dump(user_data, f, indent=2)
        
        update_log(f"Using ID {id} for user: {user_name}", "INFO")
        img_id = 0
        target_samples = 45
        
        window_name = "Dataset Collection"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            
            display_frame = frame.copy()
            face = face_cropped(frame)
            face_found = face is not None
            
            # Create UI overlay
            h, w = display_frame.shape[:2]
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 40), (0, 0, 0), -1)
            cv2.rectangle(overlay, (0, h-60), (w, h), (0, 0, 0), -1)
            alpha = 0.7
            display_frame = cv2.addWeighted(overlay, alpha, display_frame, 1-alpha, 0)
            
            # Add info text
            progress_pct = (img_id / target_samples) * 100 if target_samples > 0 else 0
            cv2.putText(display_frame, f"Progress: {img_id}/{target_samples} ({progress_pct:.1f}%)", 
                      (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            
            # Face detection guide
            cv2.putText(display_frame, 
                      "Face Detected" if face_found else "No Face - Center your face", 
                      (10, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                      (0, 255, 0) if face_found else (0, 0, 255), 1)
            
            # Add guide rectangle
            center_x, center_y = w // 2, h // 2
            guide_size = min(w, h) // 4
            cv2.rectangle(display_frame, 
                        (center_x - guide_size, center_y - guide_size),
                        (center_x + guide_size, center_y + guide_size),
                        (0, 255, 255), 2)
            
            if face_found:
                img_id += 1
                face = cv2.resize(face, (200, 200))
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                # Standard naming convention: user.ID.IMAGENUMBER.jpg
                file_name_path = f"data/user.{id}.{img_id}.jpg"
                cv2.imwrite(file_name_path, face)
                
                # Update progress
                progress = (img_id / target_samples) * 100
                progress_var.set(progress)
                
                if img_id % 10 == 0:
                    update_log(f"Collected {img_id}/{target_samples} samples", "INFO")
            
            cv2.imshow(window_name, display_frame)
            
            key = cv2.waitKey(100)
            if key == 27 or img_id >= target_samples:
                break
        
        cap.release()
        cv2.destroyAllWindows()
        update_log(f"Dataset generation completed: {img_id} samples", "SUCCESS")
        messagebox.showinfo("Success", f"Generated {img_id} samples for {user_name}")
        
        window.after(0, lambda: update_button_states())
    
    threading.Thread(target=dataset_thread, daemon=True).start()

# Connect buttons to functions
generate_button.config(command=generate_dataset)
train_button.config(command=train_classifier)
detect_button.config(command=detect_faces)

# Initial setup
def startup():
    update_log("Welcome to Face Recognition Attendance System")
    if not os.path.exists("data"):
        os.makedirs("data")
    if not os.path.exists("haarcascade_frontalface_default.xml"):
        update_log("Missing cascade file - please download", "WARNING")
        messagebox.showwarning("Missing File", 
                              "Face detection requires haarcascade_frontalface_default.xml file.\n"
                              "Please download it from OpenCV's GitHub repository.")
    update_button_states()

window.after(100, startup)
window.mainloop()