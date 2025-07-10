import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import time
import json

from video_player import VideoPlayer
from video_generator import VideoGenerator
from ui_theme import UITheme

class SpeedrunComparisonTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Speedrun Comparison Tool")
        self.root.geometry("1600x1000")
        self.root.minsize(600, 800) 
        self.set_app_icon()

        self.videos = {}
        self.video_counter = 0
        self.max_videos = 9

        self._setup_dark_title_bar()

        self.settings_file = ".\\app_settings.json"

        self.compression_settings = {
            'fps': 60,  
            'scale': 0.5,
            'codec': 'auto'
        }

        self.load_settings()
        self.gpu_available = self.check_gpu_capabilities()

        self.theme = UITheme()
        self.video_generator = VideoGenerator(self._log_operation, self._update_generation_progress)

        self.setup_gui()
        self.theme.setup_dark_theme(self.root)

        self.add_video()
        self.add_video()

    def _setup_dark_title_bar(self):
        try:
            import ctypes as ct

            DWMWA_USE_IMMERSIVE_DARK_MODE = 20
            DWMWA_USE_IMMERSIVE_DARK_MODE_BEFORE_20H1 = 19

            def set_window_attribute(hwnd, attribute, value):
                value = ct.c_int(value)
                ct.windll.dwmapi.DwmSetWindowAttribute(hwnd, attribute, ct.byref(value), ct.sizeof(value))

            self.root.update_idletasks()
            hwnd = int(self.root.wm_frame(), 16)

            try:
                set_window_attribute(hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE, 1)
            except:
                set_window_attribute(hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE_BEFORE_20H1, 1)

            print("✓ Dark title bar enabled")
        except Exception as e:
            print(f"⚠ Could not enable dark title bar: {e}")

    def set_app_icon(self):
        try:
            if os.path.exists("icon.png"):
                icon = tk.PhotoImage(file="icon.png")
                self.root.iconphoto(True, icon)
            elif os.name == 'nt' and os.path.exists("icon.ico"):
                self.root.iconbitmap("icon.ico")
            else:
                pass
        except Exception as e:
            print(f"Could not set icon: {e}")

    def check_gpu_capabilities(self):
        try:
            gpu_count = cv2.cuda.getCudaEnabledDeviceCount()
            if gpu_count > 0:
                print(f"✓ GPU acceleration available: {gpu_count} CUDA device(s)")
                return True
            else:
                print("⚠ No GPU acceleration available - using CPU")
                return False
        except Exception as e:
            print(f"⚠ GPU check failed: {e} - using CPU")
            return False

    def load_settings(self):
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, 'r') as f:
                    saved_settings = json.load(f)
                    self.compression_settings.update(saved_settings)
                print(f"Settings loaded from {self.settings_file}")
        except Exception as e:
            print(f"Could not load settings: {e}")

    def save_settings(self):
        try:
            os.makedirs(os.path.dirname(self.settings_file), exist_ok=True)
            with open(self.settings_file, 'w') as f:
                json.dump(self.compression_settings, f, indent=2)
            print(f"Settings saved to {self.settings_file}")
        except Exception as e:
            print(f"Could not save settings: {e}")

    def setup_gui(self):
        self.fps_var = tk.StringVar(value=str(int(self.compression_settings['fps'])))
        self.scale_var = tk.StringVar(value=str(self.compression_settings['scale']))
        self.codec_var = tk.StringVar(value=self.compression_settings.get('codec', 'auto'))

        main_frame = ttk.Frame(self.root, padding="10", style="Dark.TFrame")
        main_frame.pack(fill=tk.BOTH, expand=True)

        video_controls_frame = ttk.Frame(main_frame, style="Dark.TFrame")
        video_controls_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(video_controls_frame, text="⚙ Settings", 
                  command=self._open_settings_window).pack(side=tk.LEFT, padx=(0, 10))

        self.add_video_btn = ttk.Button(video_controls_frame, text="+ Add Video", 
                                    command=self.add_video)
        self.add_video_btn.pack(side=tk.LEFT, padx=5)

        ttk.Label(video_controls_frame, text="Videos:", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=(20, 5))
        self.video_count_label = ttk.Label(video_controls_frame, text="0")
        self.video_count_label.pack(side=tk.LEFT)

        self.settings_info = ttk.Label(video_controls_frame, text="Half res, 60fps, auto codec",
                                    font=("Arial", 9), foreground="gray")
        self.settings_info.pack(side=tk.RIGHT)

        self.videos_scroll_frame = ttk.Frame(main_frame, style="Dark.TFrame")
        self.videos_scroll_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.videos_scroll_frame, bg="#0f0f0f", highlightthickness=0)
        self.v_scrollbar = ttk.Scrollbar(self.videos_scroll_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas, style="Dark.TFrame")

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.v_scrollbar.set)

        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.v_scrollbar.grid(row=0, column=1, sticky="ns")

        self.videos_scroll_frame.grid_rowconfigure(0, weight=1)
        self.videos_scroll_frame.grid_columnconfigure(0, weight=1)

        def _on_mousewheel(event):
            self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")

        self.canvas.bind_all("<MouseWheel>", _on_mousewheel)

        self.canvas.bind('<Configure>', self._on_canvas_configure)

        results_frame = ttk.LabelFrame(main_frame, text="Results", padding="10", style="Dark.TLabelframe")
        results_frame.pack(fill=tk.X, pady=(10, 0))

        button_frame = ttk.Frame(results_frame, style="Dark.TFrame")
        button_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(button_frame, text="Calculate Difference", 
                command=self.calculate_difference).pack(side=tk.LEFT, padx=5)

        ttk.Button(button_frame, text="Generate Comparison Video", 
                command=self.generate_comparison_video).pack(side=tk.LEFT, padx=5)

        self.results_text = tk.Text(results_frame, height=6, width=80,
                                bg="#1a1a1a", fg="#e0e0e0", 
                                insertbackground="#e0e0e0",
                                selectbackground="#2a2a2a",
                                selectforeground="#e0e0e0")
        self.results_text.pack(fill=tk.X, pady=(0, 0))

        self._update_settings()

    def _open_settings_window(self):
        """Open the settings configuration window"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Export Settings")
        settings_window.geometry("500x400")
        settings_window.transient(self.root)
        settings_window.grab_set()
        settings_window.resizable(False, False)

        try:
            import ctypes as ct
            DWMWA_USE_IMMERSIVE_DARK_MODE = 20
            DWMWA_USE_IMMERSIVE_DARK_MODE_BEFORE_20H1 = 19

            def set_window_attribute(hwnd, attribute, value):
                value = ct.c_int(value)
                ct.windll.dwmapi.DwmSetWindowAttribute(hwnd, attribute, ct.byref(value), ct.sizeof(value))

            settings_window.update_idletasks()
            hwnd = int(settings_window.wm_frame(), 16)

            try:
                set_window_attribute(hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE, 1)
            except:
                set_window_attribute(hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE_BEFORE_20H1, 1)
        except:
            pass

        self.theme.apply_dark_theme_to_window(settings_window)

        main_frame = ttk.Frame(settings_window, padding="20", style="Dark.TFrame")
        main_frame.pack(fill=tk.BOTH, expand=True)

        title_label = ttk.Label(main_frame, text="Export Settings", 
                               font=("Arial", 14, "bold"), style="Dark.TLabel")
        title_label.pack(pady=(0, 20))

        video_frame = ttk.LabelFrame(main_frame, text="Video Quality", padding="15", style="Dark.TLabelframe")
        video_frame.pack(fill=tk.X, pady=(0, 15))

        fps_frame = ttk.Frame(video_frame, style="Dark.TFrame")
        fps_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(fps_frame, text="Output FPS:", style="Dark.TLabel").pack(side=tk.LEFT)
        fps_combo = ttk.Combobox(fps_frame, textvariable=self.fps_var, 
                                values=["20" "30", "60"], 
                                width=15, state="readonly")
        fps_combo.pack(side=tk.RIGHT)
        fps_combo.bind('<<ComboboxSelected>>', self._update_settings)

        ttk.Label(fps_frame, text="Frames per second in output video", 
                 font=("Arial", 8), foreground="gray", style="Dark.TLabel").pack(side=tk.RIGHT, padx=(0, 10))

        res_frame = ttk.Frame(video_frame, style="Dark.TFrame")
        res_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(res_frame, text="Resolution:", style="Dark.TLabel").pack(side=tk.LEFT)
        scale_combo = ttk.Combobox(res_frame, textvariable=self.scale_var, 
                                  values=["0.25", "0.5", "1.0"], 
                                  width=15, state="readonly")
        scale_combo.pack(side=tk.RIGHT)
        scale_combo.bind('<<ComboboxSelected>>', self._update_settings)

        res_desc = {"0.25": "Quarter (fastest)", "0.5": "Half (recommended)", "1.0": "Full (slow)"}
        res_label = ttk.Label(res_frame, text=res_desc.get(self.scale_var.get(), "Half (recommended)"), 
                             font=("Arial", 8), foreground="gray", style="Dark.TLabel")
        res_label.pack(side=tk.RIGHT, padx=(0, 10))

        def update_res_desc(*args):
            res_label.config(text=res_desc.get(self.scale_var.get(), "Half (recommended)"))
        self.scale_var.trace('w', update_res_desc)

        codec_frame = ttk.Frame(video_frame, style="Dark.TFrame")
        codec_frame.pack(fill=tk.X)

        ttk.Label(codec_frame, text="Codec:", style="Dark.TLabel").pack(side=tk.LEFT)
        codec_combo = ttk.Combobox(codec_frame, textvariable=self.codec_var, 
                                  values=["auto", "h264", "mp4v", "xvid"], 
                                  width=15, state="readonly")
        codec_combo.pack(side=tk.RIGHT)
        codec_combo.bind('<<ComboboxSelected>>', self._update_settings)

        ttk.Label(codec_frame, text="MP4 Codec", 
                 font=("Arial", 8), foreground="gray", style="Dark.TLabel").pack(side=tk.RIGHT, padx=(0, 10))

        perf_frame = ttk.LabelFrame(main_frame, text="Performance", padding="15", style="Dark.TLabelframe")
        perf_frame.pack(fill=tk.X, pady=(0, 20))

        gpu_status = "✓ GPU acceleration available" if self.gpu_available else "⚠ Using CPU only"
        ttk.Label(perf_frame, text=f"Hardware: {gpu_status}", 
                 font=("Arial", 9), style="Dark.TLabel").pack(anchor=tk.W)

        button_frame = ttk.Frame(main_frame, style="Dark.TFrame")
        button_frame.pack(fill=tk.X)

        ttk.Button(button_frame, text="Reset to Defaults", 
                  command=self._reset_settings).pack(side=tk.LEFT)

        ttk.Button(button_frame, text="Apply & Close", 
                  command=lambda: [self._update_settings(), settings_window.destroy()]).pack(side=tk.RIGHT)

    def _reset_settings(self):
        """Reset settings to defaults"""
        self.fps_var.set("60")
        self.scale_var.set("0.5")
        self.codec_var.set("auto")
        self._update_settings()

    def _on_canvas_configure(self, event):
        """Handle canvas resize to update video layout"""

        if hasattr(self, '_last_canvas_width'):
            if abs(event.width - self._last_canvas_width) < 10:  
                return
        self._last_canvas_width = event.width
        self.root.after_idle(self.update_layout)

    def update_layout(self):
        """Update video panel layout based on canvas width"""
        if not self.videos:
            return

        canvas_width = self.canvas.winfo_width()
        if canvas_width <= 100:  
            self.root.after(100, self.update_layout)
            return

        panel_width = 540  
        videos_per_row = max(1, canvas_width // panel_width)

        for widget in self.scrollable_frame.winfo_children():
            widget.grid_forget()

        for i, video_id in enumerate(self.videos.keys()):
            if hasattr(self, f'video_panel_{video_id}'):
                panel = getattr(self, f'video_panel_{video_id}')
                row = i // videos_per_row
                col = i % videos_per_row
                panel.grid(row=row, column=col, padx=8, pady=8, sticky=(tk.W, tk.E, tk.N, tk.S))

        current_videos_per_row = min(videos_per_row, len(self.videos))
        for col in range(current_videos_per_row):
            self.scrollable_frame.columnconfigure(col, weight=1)

        for col in range(current_videos_per_row, 10):  
            self.scrollable_frame.columnconfigure(col, weight=0)

        self.root.after_idle(lambda: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

    def rename_video(self, video_id):
        if video_id not in self.videos:
            return

        new_name = getattr(self, f'rename_entry_{video_id}').get().strip()
        if new_name and new_name != 'Enter video name...':
            panel = getattr(self, f'video_panel_{video_id}')
            panel.configure(text=f"Video {video_id}: {new_name}")
            self.videos[video_id]['custom_name'] = new_name

    def load_video(self, video_id):
        if video_id not in self.videos:
            return

        file_path = filedialog.askopenfilename(
            title=f"Select Video {video_id}",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
        )

        if file_path:
            video_data = self.videos[video_id]
            player = video_data['player']

            if player.load_video(file_path):
                seek_scale = getattr(self, f'seek_scale_{video_id}')
                seek_scale.configure(to=player.total_frames - 1)

                filename = os.path.basename(file_path)
                filename_no_ext = os.path.splitext(filename)[0]

                panel = getattr(self, f'video_panel_{video_id}')
                panel.configure(text=f"Video {video_id}: {filename_no_ext}")
                rename_entry = getattr(self, f'rename_entry_{video_id}')
                rename_entry.delete(0, tk.END)
                rename_entry.insert(0, filename_no_ext)
                rename_entry.configure(foreground='#ffffff')
                video_data['custom_name'] = filename_no_ext

                skip_ratio = max(1, int(player.fps / 60)) if player.fps > 0 else 1
                info = f"FPS: {player.fps:.1f} | Frames: {player.total_frames} | Skip: {skip_ratio}:1"
                getattr(self, f'video_info_{video_id}').configure(text=info)

                first_frame = player.get_frame_fast(0)
                if first_frame is not None:
                    frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
                    self.display_frame(video_id, 0, frame_rgb)

                self.update_frame_display(video_id, 0)
                getattr(self, f'seek_var_{video_id}').set(0)

    def toggle_play(self, video_id):
        if video_id not in self.videos:
            return

        video_data = self.videos[video_id]
        player = video_data['player']
        play_btn = getattr(self, f'play_btn_{video_id}')

        if not player.video_capture:
            return

        if player.is_playing:
            if player._pause_flag:
                player.resume_playback()
                play_btn.configure(text="⏸")
            else:
                player.pause_playback()
                play_btn.configure(text="▶")
        else:
            play_btn.configure(text="⏸")

            video_data['_displaying'] = False
            video_data['_last_info_update'] = 0

            def update_callback(frame_num, frame_bgr):
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                try:
                    self.root.after_idle(lambda: self.display_frame(video_id, frame_num, frame_rgb))
                    self.root.after_idle(lambda: getattr(self, f'seek_var_{video_id}').set(frame_num))
                except:
                    pass 

            player.start_playback(update_callback)

    def display_frame(self, video_id, frame_number, frame):
        if video_id not in self.videos:
            return

        try:
            video_data = self.videos[video_id]
            if video_data['_displaying']:
                return

            video_data['_displaying'] = True

            h, w = frame.shape[:2]
            new_width = 480
            new_height = int(new_width * h / w)

            frame_small = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

            image = Image.fromarray(frame_small)
            photo = ImageTk.PhotoImage(image)

            video_label = getattr(self, f'video_label_{video_id}')
            video_label.configure(image=photo, text="")
            video_label.image = photo

            current_time = time.time()
            last_info_update = video_data['_last_info_update']
            if current_time - last_info_update > 0.1: 
                self.update_frame_display(video_id, frame_number)
                video_data['_last_info_update'] = current_time

            video_data['_displaying'] = False
        except Exception as e:
            if video_id in self.videos:
                self.videos[video_id]['_displaying'] = False

    def update_frame_display(self, video_id, frame_number):
        if video_id not in self.videos:
            return

        video_data = self.videos[video_id]
        player = video_data['player']
        timestamp = player.get_timestamp(frame_number)

        info_text = f"{frame_number}/{player.total_frames}"
        getattr(self, f'frame_info_{video_id}').configure(text=info_text)

        time_text = f"Time: {timestamp:.2f}s"
        getattr(self, f'time_info_{video_id}').configure(text=time_text)

    def seek_frame(self, video_id, delta):
        if video_id not in self.videos:
            return

        video_data = self.videos[video_id]
        player = video_data['player']

        if not player.video_capture:
            return

        was_playing = player.is_playing and not player._pause_flag

        if player.is_playing and not player._pause_flag:
            player.pause_playback()
            play_btn = getattr(self, f'play_btn_{video_id}')
            play_btn.configure(text="▶")

        new_frame = max(0, min(player.current_frame + delta, player.total_frames - 1))
        frame = player.get_frame_fast(new_frame)

        if frame is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.display_frame(video_id, new_frame, frame_rgb)
            getattr(self, f'seek_var_{video_id}').set(new_frame)

        if was_playing:
            player.resume_playback()
            play_btn = getattr(self, f'play_btn_{video_id}')
            play_btn.configure(text="⏸")

    def on_seek(self, video_id, val):
        if video_id not in self.videos:
            return

        video_data = self.videos[video_id]
        player = video_data['player']

        if not player.video_capture or not player.video_path:
            return

        try:
            frame_number = int(float(val))
            frame_number = max(0, min(frame_number, player.total_frames - 1))

            was_playing = player.is_playing
            if was_playing:
                player.stop_playback()

            frame = player.get_frame_fast(frame_number)
            if frame is not None:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.display_frame(video_id, frame_number, frame_rgb)
                self.update_frame_display(video_id, frame_number)

            if was_playing:
                play_btn = getattr(self, f'play_btn_{video_id}')
                play_btn.configure(text="▶")

        except Exception as e:
            print(f"Seek error: {e}") 
            pass

    def reset_marks(self, video_id):
        if video_id not in self.videos:
            return
        video_data = self.videos[video_id]
        video_data['start_frame'] = 0
        video_data['end_frame'] = 0
        getattr(self, f'marked_info_{video_id}').configure(text="Start:      - | End:      -")

    def jump_to_mark(self, video_id, mark_type):
        if video_id not in self.videos:
            return

        video_data = self.videos[video_id]
        mark = video_data['start_frame'] if mark_type == 'start' else video_data['end_frame']
        player = video_data['player']

        if not player.video_capture:
            messagebox.showwarning("Warning", "No video loaded.")
            return

        if mark == 0 and mark_type == 'end':
            messagebox.showwarning("Warning", f"No {mark_type} frame marked.")
            return

        if 0 <= mark < player.total_frames:
            if player.is_playing:
                self.toggle_play(video_id)

            frame = player.get_frame_fast(mark)
            if frame is not None:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.display_frame(video_id, mark, frame_rgb)
                getattr(self, f'seek_var_{video_id}').set(mark)
        else:
            messagebox.showwarning("Warning", f"Invalid {mark_type} frame: {mark}")

    def mark_frame(self, video_id, mark_type):
        if video_id not in self.videos:
            return

        video_data = self.videos[video_id]
        player = video_data['player']
        current_frame = player.current_frame

        if mark_type == 'start':
            video_data['start_frame'] = current_frame
        else:
            video_data['end_frame'] = current_frame

        start_frame = video_data['start_frame']
        end_frame = video_data['end_frame']
        info_text = f"Start: {start_frame:>6} | End: {end_frame:>6}"
        getattr(self, f'marked_info_{video_id}').configure(text=info_text)

    def calculate_difference(self):
        loaded_videos = {vid: data for vid, data in self.videos.items() if data['player'].video_capture}

        if len(loaded_videos) < 2:
            messagebox.showerror("Error", "Please load at least 2 videos first.")
            return

        durations = {}
        for video_id, video_data in loaded_videos.items():
            duration = self.calculate_duration(video_id)
            if duration is None:
                messagebox.showerror("Error", f"Please mark start and end frames for Video {video_id}.")
                return
            durations[video_id] = duration

        results = "=== COMPARISON RESULTS ===\n\n"

        video_list = list(durations.keys())
        for i, video_id in enumerate(video_list):
            video_name = loaded_videos[video_id]['custom_name']
            start_frame = loaded_videos[video_id]['start_frame']
            end_frame = loaded_videos[video_id]['end_frame']
            results += f"{video_name}: {durations[video_id]:.3f}s ({start_frame} → {end_frame})\n"

        results += "\nDIFFERENCES:\n"
        base_video = video_list[0]
        base_duration = durations[base_video]
        base_name = loaded_videos[base_video]['custom_name']

        for video_id in video_list[1:]:
            video_name = loaded_videos[video_id]['custom_name']
            time_diff = durations[video_id] - base_duration
            status = "SLOWER" if time_diff > 0 else "FASTER" if time_diff < 0 else "SAME TIME"
            results += f"{video_name} vs {base_name}: {time_diff:.3f}s ({status})\n"

        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, results)

    def calculate_duration(self, video_id):
        if video_id not in self.videos:
            return None

        video_data = self.videos[video_id]
        player = video_data['player']
        start_frame = video_data['start_frame']
        end_frame = video_data['end_frame']

        if start_frame >= end_frame:
            return None

        start_time = player.get_timestamp(start_frame)
        end_time = player.get_timestamp(end_frame)
        return end_time - start_time

    def generate_comparison_video(self):
        loaded_videos = {vid: data for vid, data in self.videos.items() if data['player'].video_capture}

        if len(loaded_videos) < 2:
            messagebox.showerror("Error", "Please load at least 2 videos first.")
            return

        invalid_videos = []
        for video_id, video_data in loaded_videos.items():
            if video_data['start_frame'] >= video_data['end_frame']:
                invalid_videos.append(video_id)

        if invalid_videos:
            video_names = [loaded_videos[vid]['custom_name'] for vid in invalid_videos]
            messagebox.showerror("Error", f"Please mark valid start and end frames for: {', '.join(video_names)}.")
            return

        output_path = filedialog.asksaveasfilename(
            title="Save Comparison Video",
            defaultextension=".mp4",
            filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")]
        )

        if not output_path:
            return

        self._create_detailed_progress_window(output_path, self.compression_settings)

    def _create_detailed_progress_window(self, output_path, compression_settings):
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Video Generation - Detailed View")
        progress_window.geometry("800x600")
        progress_window.transient(self.root)
        progress_window.grab_set()

        self.video_generator.set_cancel_flag(False)
        self.video_generator.set_pause_flag(False)

        self._current_progress_window = progress_window

        try:
            import ctypes as ct
            DWMWA_USE_IMMERSIVE_DARK_MODE = 20
            DWMWA_USE_IMMERSIVE_DARK_MODE_BEFORE_20H1 = 19

            def set_window_attribute(hwnd, attribute, value):
                value = ct.c_int(value)
                ct.windll.dwmapi.DwmSetWindowAttribute(hwnd, attribute, ct.byref(value), ct.sizeof(value))

            progress_window.update_idletasks()
            hwnd = int(progress_window.wm_frame(), 16)

            try:
                set_window_attribute(hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE, 1)
            except:
                set_window_attribute(hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE_BEFORE_20H1, 1)
        except:
            pass 

        self.theme.apply_dark_theme_to_window(progress_window)

        main_frame = ttk.Frame(progress_window, padding="10", style="Dark.TFrame")
        main_frame.pack(fill=tk.BOTH, expand=True)

        title_label = ttk.Label(main_frame, text="Video Generation Progress", 
                               font=("Arial", 14, "bold"), style="Dark.TLabel")
        title_label.pack(pady=(0, 10))

        info_frame = ttk.LabelFrame(main_frame, text="Generation Info", padding="10", style="Dark.TLabelframe")
        info_frame.pack(fill=tk.X, pady=(0, 10))

        self.gen_status_label = ttk.Label(info_frame, text="Initializing...", 
                                         font=("Arial", 10, "bold"))
        self.gen_status_label.pack(anchor=tk.W)

        self.gen_progress_bar = ttk.Progressbar(info_frame, mode='determinate')
        self.gen_progress_bar.pack(fill=tk.X, pady=(5, 0))

        progress_info_frame = ttk.Frame(info_frame, style="Dark.TFrame")
        progress_info_frame.pack(fill=tk.X, pady=(5, 0))

        self.gen_progress_percent = ttk.Label(progress_info_frame, text="0%")
        self.gen_progress_percent.pack(side=tk.LEFT)

        self.gen_eta_label = ttk.Label(progress_info_frame, text="ETA: Calculating...")
        self.gen_eta_label.pack(side=tk.RIGHT)

        speed_frame = ttk.Frame(info_frame, style="Dark.TFrame")
        speed_frame.pack(fill=tk.X, pady=(5, 0))

        self.gen_speed_label = ttk.Label(speed_frame, text="Speed: - fps")
        self.gen_speed_label.pack(side=tk.LEFT)

        self.gen_memory_label = ttk.Label(speed_frame, text="Memory: -")
        self.gen_memory_label.pack(side=tk.RIGHT)

        log_frame = ttk.LabelFrame(main_frame, text="Detailed Operations Log", padding="10", style="Dark.TLabelframe")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        log_text_frame = ttk.Frame(log_frame, style="Dark.TFrame")
        log_text_frame.pack(fill=tk.BOTH, expand=True)

        self.gen_log_text = tk.Text(log_text_frame, height=20, width=80, 
                                   font=("Courier", 9), wrap=tk.WORD,
                                   bg="#1a1a1a", fg="#e0e0e0", 
                                   insertbackground="#e0e0e0",
                                   selectbackground="#2a2a2a",
                                   selectforeground="#e0e0e0")
        log_scrollbar = ttk.Scrollbar(log_text_frame, orient=tk.VERTICAL, 
                                     command=self.gen_log_text.yview)
        self.gen_log_text.configure(yscrollcommand=log_scrollbar.set)

        self.gen_log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        button_frame = ttk.Frame(main_frame, style="Dark.TFrame")
        button_frame.pack(fill=tk.X)

        self.gen_cancel_btn = ttk.Button(button_frame, text="Cancel Generation", 
                                        command=lambda: self._cancel_generation())
        self.gen_cancel_btn.pack(side=tk.LEFT, padx=(0, 10))

        self.gen_pause_btn = ttk.Button(button_frame, text="Pause", 
                                       command=self._toggle_generation_pause)
        self.gen_pause_btn.pack(side=tk.LEFT, padx=(0, 10))

        ttk.Button(button_frame, text="Clear Log", 
                  command=lambda: self.gen_log_text.delete(1.0, tk.END)).pack(side=tk.LEFT)

        self.auto_scroll_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(button_frame, text="Auto-scroll", 
                       variable=self.auto_scroll_var).pack(side=tk.RIGHT)

        self._generation_start_time = time.time()

        def generate_thread():
            try:
                loaded_videos = {vid: data for vid, data in self.videos.items() if data['player'].video_capture}
                self.video_generator.generate_comparison_video(output_path, loaded_videos, compression_settings)
            except Exception as e:
                self._log_operation(f"ERROR: {str(e)}", "error")
                self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to generate video: {str(e)}"))
            finally:
                self.root.after(0, lambda: self._finalize_generation(progress_window, output_path))

        thread = threading.Thread(target=generate_thread)
        thread.daemon = True
        thread.start()

    def _toggle_generation_pause(self):
        current_pause = getattr(self.video_generator, '_pause_generation', False)
        self.video_generator.set_pause_flag(not current_pause)
        if not current_pause:
            self.gen_pause_btn.configure(text="Resume")
            self._log_operation("Generation PAUSED by user", "warning")
        else:
            self.gen_pause_btn.configure(text="Pause")
            self._log_operation("Generation RESUMED by user", "info")

    def _log_operation(self, message, level="info"):
        timestamp = time.strftime("%H:%M:%S")

        colors = {
            "info": "black",
            "success": "dark green", 
            "warning": "orange",
            "error": "red",
            "debug": "blue"
        }

        log_entry = f"[{timestamp}] {message}\n"

        def update_log():
            if hasattr(self, 'gen_log_text'):
                self.gen_log_text.insert(tk.END, log_entry)
                self.gen_log_text.tag_add(level, f"end-{len(log_entry)}c", "end-1c")
                self.gen_log_text.tag_config(level, foreground=colors.get(level, "black"))

                if hasattr(self, 'auto_scroll_var') and self.auto_scroll_var.get():
                    self.gen_log_text.see(tk.END)

        self.root.after(0, update_log)

    def _update_generation_progress(self, current, total, operation="", extra_info=""):
        if total > 0:
            progress = (current / total) * 100
            elapsed = time.time() - self._generation_start_time

            def update_ui():
                try:
                    if hasattr(self, 'gen_progress_bar'):
                        self.gen_progress_bar['maximum'] = total
                        self.gen_progress_bar['value'] = current
                        self.gen_progress_percent.configure(text=f"{progress:.1f}%")

                        if operation:
                            status_text = f"Status: {operation}"
                            if extra_info:
                                status_text += f" | {extra_info}"
                            self.gen_status_label.configure(text=status_text)

                        if current > 0 and progress > 0:
                            eta_seconds = (elapsed / current) * (total - current)
                            eta_text = f"ETA: {int(eta_seconds//60)}m {int(eta_seconds%60)}s"
                        else:
                            eta_text = "ETA: Calculating..."
                        self.gen_eta_label.configure(text=eta_text)

                        if elapsed > 0:
                            fps = current / elapsed
                            self.gen_speed_label.configure(text=f"Speed: {fps:.1f} fps")
                except:
                    pass 

            self.root.after(0, update_ui)

    def _cancel_generation(self):
        """Cancel the generation and close window if generation is complete"""
        self.video_generator.set_cancel_flag(True)
        if hasattr(self, '_current_progress_window') and hasattr(self, 'gen_cancel_btn'):
            if self.gen_cancel_btn.cget('text') == 'Close':
                self._current_progress_window.destroy()

    def _finalize_generation(self, progress_window, output_path):
        if hasattr(self, 'gen_cancel_btn'):
            self.gen_cancel_btn.configure(text="Close")
        if hasattr(self, 'gen_pause_btn'):
            self.gen_pause_btn.configure(state="disabled")

        if not getattr(self.video_generator, '_cancel_generation', False) and os.path.exists(output_path):
            messagebox.showinfo("Success", 
                f"Comparison video generated successfully!\n\n"
                f"Saved to: {output_path}\n\n"
                )

    def add_video(self):
        if len(self.videos) >= self.max_videos:
            messagebox.showwarning("Limit Reached", f"Maximum of {self.max_videos} videos allowed.")
            return

        self.video_counter += 1
        video_id = self.video_counter

        video_data = {
            'player': VideoPlayer(),
            'start_frame': 0,
            'end_frame': 0,
            'current_frame': tk.IntVar(value=0),
            '_displaying': False,
            '_last_info_update': 0,
            'custom_name': f'Video {video_id}',
            'audio_enabled': False
        }

        self.videos[video_id] = video_data

        self.create_video_panel(video_id)
        self.update_video_count()
        self.update_layout()

        if len(self.videos) >= self.max_videos:
            self.add_video_btn.configure(state="disabled")

    def remove_video(self, video_id):
        if video_id in self.videos:
            video_data = self.videos[video_id]
            video_data['player'].close()

            if hasattr(self, f'video_panel_{video_id}'):
                panel = getattr(self, f'video_panel_{video_id}')
                panel.destroy()
                delattr(self, f'video_panel_{video_id}')

            del self.videos[video_id]
            self.update_video_count()
            self.update_layout()

            if len(self.videos) < self.max_videos:
                self.add_video_btn.configure(state="normal")

    def update_video_count(self):
        self.video_count_label.configure(text=str(len(self.videos)))

    def update_layout(self):
        """Update video panel layout based on available screen space"""
        if not self.videos:
            return

        canvas_width = self.canvas.winfo_width()
        if canvas_width <= 100:  
            self.root.after(100, self.update_layout)
            return

        panel_width = 540  
        videos_per_row = max(1, canvas_width // panel_width)

        for widget in self.scrollable_frame.winfo_children():
            widget.grid_forget()

        for i, video_id in enumerate(self.videos.keys()):
            if hasattr(self, f'video_panel_{video_id}'):
                panel = getattr(self, f'video_panel_{video_id}')
                row = i // videos_per_row
                col = i % videos_per_row
                panel.grid(row=row, column=col, padx=8, pady=8, sticky=(tk.W, tk.E, tk.N, tk.S))

        current_videos_per_row = min(videos_per_row, len(self.videos))
        for col in range(current_videos_per_row):
            self.scrollable_frame.columnconfigure(col, weight=1)

        for col in range(current_videos_per_row, 10):  
            self.scrollable_frame.columnconfigure(col, weight=0)

        self.root.after_idle(lambda: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

    def create_video_panel(self, video_id):
        panel = ttk.LabelFrame(self.scrollable_frame, text=f"Video {video_id}", 
                          padding="10", style="Dark.TLabelframe")
        setattr(self, f'video_panel_{video_id}', panel)

        load_frame = ttk.Frame(panel, style="Dark.TFrame")
        load_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(load_frame, text="Load Video", 
                command=lambda: self.load_video(video_id)).pack(side=tk.LEFT, padx=(0, 5))

        remove_btn = ttk.Button(load_frame, text="✕", width=3,
                            command=lambda: self.remove_video(video_id))
        remove_btn.pack(side=tk.RIGHT, padx=(5, 0))

        audio_var = tk.BooleanVar(value=False)
        audio_checkbox = ttk.Checkbutton(load_frame, text="Audio", 
                                        variable=audio_var,
                                        command=lambda: self.toggle_audio(video_id, audio_var.get()))
        audio_checkbox.pack(side=tk.RIGHT, padx=(5, 0))
        setattr(self, f'audio_var_{video_id}', audio_var)

        rename_entry = ttk.Entry(load_frame, width=20, font=("Arial", 8))
        rename_entry.pack(side=tk.LEFT, padx=(0, 5), fill=tk.X, expand=True)
        setattr(self, f'rename_entry_{video_id}', rename_entry)

        def on_entry_click(event):
            if rename_entry.get() == 'Enter video name...':
                rename_entry.delete(0, tk.END)
                rename_entry.configure(foreground='#ffffff')

        def on_focusout(event):
            if rename_entry.get() == '':
                rename_entry.insert(0, 'Enter video name...')
                rename_entry.configure(foreground='#888888')
            else:
                self.rename_video(video_id)

        rename_entry.bind('<FocusIn>', on_entry_click)
        rename_entry.bind('<FocusOut>', on_focusout)

        rename_entry.insert(0, 'Enter video name...')
        rename_entry.configure(foreground='#888888')

        video_info = ttk.Label(panel, text="No video loaded", font=("Arial", 9), anchor="center")
        video_info.pack(fill=tk.X, pady=(0, 5))
        setattr(self, f'video_info_{video_id}', video_info)

        video_frame = ttk.Frame(panel, height=300, style="Dark.TFrame")
        video_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        video_frame.pack_propagate(False)

        video_label = ttk.Label(video_frame, text="Load a video to see preview", anchor="center")
        video_label.pack(fill=tk.BOTH, expand=True)
        setattr(self, f'video_label_{video_id}', video_label)

        seek_frame = ttk.Frame(panel, style="Dark.TFrame")
        seek_frame.pack(fill=tk.X, pady=(0, 10))

        seek_var = tk.IntVar(value=0)
        seek_scale = ttk.Scale(seek_frame, from_=0, to=100, orient=tk.HORIZONTAL, 
                            variable=seek_var,
                            command=lambda val: self.on_seek(video_id, val))
        seek_scale.pack(fill=tk.X, padx=(0, 10), side=tk.LEFT, expand=True)
        setattr(self, f'seek_var_{video_id}', seek_var)
        setattr(self, f'seek_scale_{video_id}', seek_scale)

        frame_info = ttk.Label(seek_frame, text="0/0", font=("Courier", 9), width=12)
        frame_info.pack(side=tk.RIGHT)
        setattr(self, f'frame_info_{video_id}', frame_info)

        controls = ttk.Frame(panel, style="Dark.TFrame")
        controls.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(controls, text="<<", width=3,
                command=lambda: self.seek_frame(video_id, -10)).pack(side=tk.LEFT, padx=1)
        ttk.Button(controls, text="<", width=3,
                command=lambda: self.seek_frame(video_id, -1)).pack(side=tk.LEFT, padx=1)
        play_btn = ttk.Button(controls, text="▶", width=3,
                            command=lambda: self.toggle_play(video_id))
        play_btn.pack(side=tk.LEFT, padx=2)
        setattr(self, f'play_btn_{video_id}', play_btn)
        ttk.Button(controls, text=">", width=3,
                command=lambda: self.seek_frame(video_id, 1)).pack(side=tk.LEFT, padx=1)
        ttk.Button(controls, text=">>", width=3,
                command=lambda: self.seek_frame(video_id, 10)).pack(side=tk.LEFT, padx=1)

        mark_frame = ttk.Frame(panel, style="Dark.TFrame")
        mark_frame.pack(fill=tk.X, pady=(0, 10))  

        ttk.Button(mark_frame, text="Mark Start", 
                command=lambda: self.mark_frame(video_id, 'start')).pack(side=tk.LEFT, padx=2)
        ttk.Button(mark_frame, text="Mark End", 
                command=lambda: self.mark_frame(video_id, 'end')).pack(side=tk.LEFT, padx=2)

        ttk.Button(mark_frame, text="Jump Start", width=12,
                command=lambda: self.jump_to_mark(video_id, 'start')).pack(side=tk.LEFT, padx=2)
        ttk.Button(mark_frame, text="Jump End", width=12,
                command=lambda: self.jump_to_mark(video_id, 'end')).pack(side=tk.LEFT, padx=2)

        marked_info = ttk.Label(mark_frame, text="Start: 0 | End: 0", font=("Courier", 8), width=32)
        marked_info.pack(side=tk.RIGHT)
        setattr(self, f'marked_info_{video_id}', marked_info)

        time_info = ttk.Label(panel, text="Time: 0.00s", font=("Courier", 9), anchor="center")
        time_info.pack(fill=tk.X, pady=(5, 0))  
        setattr(self, f'time_info_{video_id}', time_info)

    def on_closing(self):
        self.save_settings()

        if hasattr(self.video_generator, '_cancel_generation'):
            self.video_generator.set_cancel_flag(True)
            self.video_generator.set_pause_flag(False)

        for video_id, video_data in self.videos.items():
            if 'player' in video_data and video_data['player']:
                video_data['player'].close()

        self.root.destroy()

    def toggle_audio(self, video_id, enabled):
        """Toggle audio enable/disable for a video"""
        if video_id in self.videos:
            self.videos[video_id]['audio_enabled'] = enabled
            if enabled:
                self._log_audio_status(video_id)
            else:
                if hasattr(self, 'gen_log_text'):
                    video_name = self.videos[video_id]['custom_name']
                    self._log_operation(f"Audio disabled for: {video_name}", "info")

    def _log_audio_status(self, video_id):
        """Log which video has audio enabled"""
        if hasattr(self, 'gen_log_text'):
            video_name = self.videos[video_id]['custom_name']
            self._log_operation(f"Audio enabled for: {video_name}", "info")

    def _update_settings(self, event=None):
        try:
            self.compression_settings['fps'] = float(self.fps_var.get())
            self.compression_settings['scale'] = float(self.scale_var.get())
            if hasattr(self, 'codec_var'):
                self.compression_settings['codec'] = self.codec_var.get()
            self.save_settings()
            scale_text = {0.25: "Quarter", 0.5: "Half", 1.0: "Full"}[self.compression_settings['scale']]
            fps_text = f"{int(self.compression_settings['fps'])}fps"
            codec_text = self.compression_settings.get('codec', 'auto')
            info_text = f"{scale_text} res, {fps_text}, {codec_text} codec"
            if hasattr(self, 'settings_info'):
                self.settings_info.configure(text=info_text)
        except (ValueError, KeyError):
            pass

    def on_closing(self):
        self.save_settings()

        if hasattr(self.video_generator, '_cancel_generation'):
            self.video_generator.set_cancel_flag(True)
            self.video_generator.set_pause_flag(False)

        for video_id, video_data in self.videos.items():
            if 'player' in video_data and video_data['player']:
                video_data['player'].close()

        self.root.destroy()

    def toggle_audio(self, video_id, enabled):
        """Toggle audio enable/disable for a video"""
        if video_id in self.videos:
            self.videos[video_id]['audio_enabled'] = enabled
            if enabled:
                self._log_audio_status(video_id)
            else:
                if hasattr(self, 'gen_log_text'):
                    video_name = self.videos[video_id]['custom_name']
                    self._log_operation(f"Audio disabled for: {video_name}", "info")

    def _log_audio_status(self, video_id):
        """Log which video has audio enabled"""
        if hasattr(self, 'gen_log_text'):
            video_name = self.videos[video_id]['custom_name']
            self._log_operation(f"Audio enabled for: {video_name}", "info")

    def _update_settings(self, event=None):
        try:
            self.compression_settings['fps'] = float(self.fps_var.get())
            self.compression_settings['scale'] = float(self.scale_var.get())
            if hasattr(self, 'codec_var'):
                self.compression_settings['codec'] = self.codec_var.get()
            self.save_settings()
            scale_text = {0.25: "Quarter", 0.5: "Half", 1.0: "Full"}[self.compression_settings['scale']]
            fps_text = f"{int(self.compression_settings['fps'])}fps"
            codec_text = self.compression_settings.get('codec', 'auto')
            info_text = f"{scale_text} res, {fps_text}, {codec_text} codec"
            if hasattr(self, 'settings_info'):
                self.settings_info.configure(text=info_text)
        except (ValueError, KeyError):
            pass

if __name__ == "__main__":
    root = tk.Tk()
    app = SpeedrunComparisonTool(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()