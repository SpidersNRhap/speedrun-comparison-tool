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
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import queue
import multiprocessing as mp
from functools import partial
import json
import math

class VideoPlayer:
    
    def __init__(self):
        self.video_capture = None
        self.total_frames = 0
        self.fps = 0
        self.current_frame = 0
        self.video_path = None
        self.is_playing = False
        self.play_thread = None
        self.target_fps = 60   
        self._stop_flag = False
        self._pause_flag = False
        self._pause_start_time = 0
        self._total_pause_time = 0
        
        self._frame_cache = {}
        self._cache_size_limit = 50
        self._last_frame_time = 0
        
        self.frame_buffer = queue.Queue(maxsize=10)  
        self.last_render_time = 0
        self.frames_dropped = 0
        self.actual_fps = 0
        self.gpu_available = False
        
        self._try_gpu_acceleration()
        
    def load_video(self, video_path):
        try:
            if self.video_capture:
                self.video_capture.release()
            
            self._clear_cache()
            
            backends_to_try = [
                (cv2.CAP_FFMPEG, "FFmpeg"),
                (cv2.CAP_ANY, "Default"), 
                (cv2.CAP_MSMF, "Media Foundation")
            ]
            
            for backend, backend_name in backends_to_try:
                try:
                    self.video_capture = cv2.VideoCapture(video_path, backend)
                    
                    if self.video_capture.isOpened():
                        self.video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        
                        self.total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
                        self.fps = self.video_capture.get(cv2.CAP_PROP_FPS)
                        self.current_frame = 0
                        self.video_path = video_path
                        
                        ret, test_frame = self.video_capture.read()
                        if ret and test_frame is not None:
                            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            print(f"Successfully loaded video using {backend_name} backend")
                            return True
                    
                    if self.video_capture:
                        self.video_capture.release()
                        
                except Exception:
                    if self.video_capture:
                        try:
                            self.video_capture.release()
                        except:
                            pass
                    continue
            
            raise ValueError("Could not open video file with any backend")
            
        except Exception as e:
            if self.video_capture:
                try:
                    self.video_capture.release()
                except:
                    pass
                self.video_capture = None
            messagebox.showerror("Error", f"Failed to load video: {str(e)}")
            return False
    
    def get_frame_fast(self, frame_number):
        if not self.video_capture or not self.video_path:
            return None

        cached_frame = self._get_cached_frame(frame_number)
        if cached_frame is not None:
            self.current_frame = frame_number
            return cached_frame

        try:
            if frame_number == 0:
                self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.video_capture.read()
                if ret and frame is not None:
                    self.current_frame = 0
                    self._cache_frame(0, frame)
                    return frame
                self.video_capture.set(cv2.CAP_PROP_POS_MSEC, 0)
                ret, frame = self.video_capture.read()
                if ret and frame is not None:
                    self.current_frame = 0
                    self._cache_frame(0, frame)
                    return frame
            
            if self.total_frames > 5000 and frame_number > 10:
                timestamp = frame_number / self.fps if self.fps > 0 else 0
                timestamp_ms = timestamp * 1000
                
                try:
                    self.video_capture.set(cv2.CAP_PROP_POS_MSEC, timestamp_ms)
                    ret, frame = self.video_capture.read()
                    if ret and frame is not None:
                        self.current_frame = frame_number
                        self._cache_frame(frame_number, frame)
                        return frame
                except:
                    pass
            
            if self._safe_seek(frame_number):
                ret, frame = self.video_capture.read()
                
                if ret and frame is not None:
                    self.current_frame = frame_number
                    self._cache_frame(frame_number, frame)
                    return frame
            
            if self.video_capture:
                self.video_capture.release()
            
            backends_to_try = [
                (cv2.CAP_FFMPEG, "FFmpeg"),
                (cv2.CAP_ANY, "Default"), 
                (cv2.CAP_MSMF, "Media Foundation")
            ]
            
            for backend, backend_name in backends_to_try:
                try:
                    self.video_capture = cv2.VideoCapture(self.video_path, backend)
                    if self.video_capture.isOpened():
                        self.video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        
                        if self.total_frames > 5000:
                            timestamp = frame_number / self.fps if self.fps > 0 else 0
                            timestamp_ms = timestamp * 1000
                            try:
                                self.video_capture.set(cv2.CAP_PROP_POS_MSEC, timestamp_ms)
                            except:
                                self._safe_seek(frame_number)
                        else:
                            self._safe_seek(frame_number)
                            
                        ret, frame = self.video_capture.read()
                        if ret and frame is not None:
                            self.current_frame = frame_number
                            self._cache_frame(frame_number, frame)
                            return frame
                        break
                except:
                    continue
            
            return None
            
        except Exception:
            return None
    
    def start_playback(self, update_callback):
        if self.is_playing or not self.video_capture:
            return
            
        self.is_playing = True
        self._stop_flag = False
        self._pause_flag = False
        self._total_pause_time = 0
        self.play_thread = threading.Thread(target=self._play_loop, args=(update_callback,))
        self.play_thread.daemon = True
        self.play_thread.start()
    
    def stop_playback(self):
        self._stop_flag = True
        self._pause_flag = False
        self.is_playing = False
        if self.play_thread:
            self.play_thread.join(timeout=0.1)
    
    def pause_playback(self):
        """Pause the current playback without stopping the thread"""
        if self.is_playing and not self._pause_flag:
            self._pause_flag = True
            self._pause_start_time = time.perf_counter()
    
    def resume_playback(self):
        """Resume paused playback"""
        if self.is_playing and self._pause_flag:
            if self._pause_start_time > 0:
                pause_duration = time.perf_counter() - self._pause_start_time
                self._total_pause_time += pause_duration
            self._pause_flag = False
            self._pause_start_time = 0
    
    def _play_loop(self, update_callback):
        if self.fps <= 0:
            return
            
        video_frame_time = 1.0 / self.fps  
        max_display_fps = 60  
        display_frame_time = 1.0 / max_display_fps
        
        skip_ratio = max(1, int(self.fps / max_display_fps))
        
        playback_start_time = time.perf_counter()
        last_display_time = 0
        frames_processed = 0
        frames_displayed = 0
        last_seek_time = 0
        seek_cooldown = 0.1
        
        expected_frame = self.current_frame
        frames_since_seek = 0
        max_frames_without_sync = 50
        
        while not self._stop_flag and self.current_frame < self.total_frames - 1:
            if self._pause_flag:
                time.sleep(0.1)
                continue
                
            loop_start = time.perf_counter()
            
            try:
                raw_elapsed = loop_start - playback_start_time
                adjusted_elapsed = raw_elapsed - self._total_pause_time
                
                target_frame = int(adjusted_elapsed * self.fps)
                target_frame = min(target_frame, self.total_frames - 1)
                
                if frames_since_seek > max_frames_without_sync:
                    try:
                        actual_pos = int(self.video_capture.get(cv2.CAP_PROP_POS_FRAMES))
                        tolerance = 10 if expected_frame < 50 else 5
                        if abs(actual_pos - expected_frame) > tolerance:
                            expected_frame = actual_pos
                        frames_since_seek = 0
                    except:
                        pass
                
                time_since_last_seek = loop_start - last_seek_time
                frame_diff = target_frame - expected_frame
                
                if (frame_diff > 5 and
                    time_since_last_seek > seek_cooldown and 
                    target_frame > expected_frame): 
                    
                    if self._safe_seek(target_frame):
                        last_seek_time = loop_start
                        expected_frame = target_frame
                        frames_since_seek = 0
                    
                ret, frame = self.video_capture.read()
                if not ret or frame is None:
                    if time_since_last_seek > 1.0:
                        try:
                            if self.video_path and self.video_capture:
                                self.video_capture.release()
                                self.video_capture = cv2.VideoCapture(self.video_path, cv2.CAP_FFMPEG)
                                if not self.video_capture.isOpened():
                                    self.video_capture = cv2.VideoCapture(self.video_path, cv2.CAP_ANY)
                                if self.video_capture.isOpened():
                                    self.video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                                    if self._safe_seek(expected_frame):
                                        ret, frame = self.video_capture.read()
                                        last_seek_time = loop_start
                                if not ret:
                                    break
                        except:
                            break
                    else:
                        break
                    
                if ret and frame is not None:
                    if frames_since_seek > 20:
                        try:
                            actual_frame = int(self.video_capture.get(cv2.CAP_PROP_POS_FRAMES))
                            expected_frame = actual_frame
                            self.current_frame = actual_frame
                            frames_since_seek = 0
                        except:
                            expected_frame += 1
                            self.current_frame = expected_frame
                    else:
                        expected_frame += 1
                        self.current_frame = expected_frame
                    frames_since_seek += 1
                else:
                    pass
                
                current_time = time.perf_counter()
                
                frames_elapsed_time = frames_processed / self.fps
                actual_elapsed_time = current_time - playback_start_time - self._total_pause_time
                
                time_ahead = actual_elapsed_time - frames_elapsed_time
                if time_ahead < 0:
                    sleep_time = abs(time_ahead)
                    sleep_time = min(sleep_time, 1.0 / self.fps)
                    if sleep_time > 0.001:
                        time.sleep(sleep_time)
                
                current_time = time.perf_counter()
                
                frames_processed += 1
                should_display = False
                
                if self.fps <= max_display_fps:
                    should_display = True
                else:
                    time_since_last_display = current_time - last_display_time
                    if (time_since_last_display >= display_frame_time or 
                        frames_displayed == 0 or 
                        frames_processed % skip_ratio == 0):
                        should_display = True
                
                if should_display and frame is not None:
                    try:
                        update_callback(self.current_frame, frame)
                        last_display_time = current_time
                        frames_displayed += 1
                    except:
                        pass
                else:
                    self.frames_dropped += 1
                
                if frames_processed % 60 == 0:
                    total_elapsed = current_time - playback_start_time - self._total_pause_time
                    if total_elapsed > 0:
                        self.actual_fps = frames_displayed / total_elapsed
                        if self.actual_fps > self.fps * 1.02:
                            print(f"Warning: Playback running at {self.actual_fps:.1f} FPS (source: {self.fps:.1f} FPS)")
                        
            except Exception as e:
                print(f"Playback error (continuing): {e}")
                time.sleep(0.01)
                continue
        
        self.is_playing = False
    
    def get_timestamp(self, frame_number):
        return frame_number / self.fps if self.fps > 0 else 0
    
    def close(self):
        self.stop_playback()
        self._clear_cache()
        if self.video_capture:
            self.video_capture.release()
    
    def _try_gpu_acceleration(self):
        try:
            backends = [
                (cv2.CAP_DSHOW, "DirectShow"),
                (cv2.CAP_MSMF, "Media Foundation"), 
                (cv2.CAP_FFMPEG, "FFmpeg"),
            ]
            
            for backend, name in backends:
                test_cap = cv2.VideoCapture()
                test_cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
                if test_cap.isOpened():
                    self.gpu_available = True
                    print(f"GPU acceleration available via {name}")
                    break
                test_cap.release()
        except:
            self.gpu_available = False

    def _safe_seek(self, frame_number):
        try:
            frame_number = max(0, min(frame_number, self.total_frames - 1))
            
            if frame_number == 0:
                success = self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                if not success:
                    self.video_capture.set(cv2.CAP_PROP_POS_MSEC, 0)
                return True
            
            if self.total_frames > 5000:
                success = self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                if success:
                    if frame_number > 10:
                        try:
                            actual_pos = int(self.video_capture.get(cv2.CAP_PROP_POS_FRAMES))
                            if abs(actual_pos - frame_number) > 20:
                                timestamp = frame_number / self.fps if self.fps > 0 else 0
                                timestamp_ms = timestamp * 1000
                                self.video_capture.set(cv2.CAP_PROP_POS_MSEC, timestamp_ms)
                        except:
                            pass
                return success
            else:
                success = self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                
                if success and frame_number > 5:
                    try:
                        actual_pos = int(self.video_capture.get(cv2.CAP_PROP_POS_FRAMES))
                        tolerance = 10 if frame_number < 50 else 5
                        if abs(actual_pos - frame_number) > tolerance:
                            current_pos = actual_pos
                            while current_pos < frame_number and current_pos < self.total_frames - 1:
                                ret, _ = self.video_capture.read()
                                if not ret:
                                    break
                                current_pos += 1
                    except:
                        pass
                        
                return True
                
        except Exception as e:
            print(f"Seek error: {e}")
            return False

    def _cache_frame(self, frame_number, frame):
        """Cache a frame for quick access"""
        if len(self._frame_cache) >= self._cache_size_limit:
            oldest_key = min(self._frame_cache.keys())
            del self._frame_cache[oldest_key]
        self._frame_cache[frame_number] = frame.copy()
    
    def _get_cached_frame(self, frame_number):
        """Get a frame from cache if available"""
        return self._frame_cache.get(frame_number)
    
    def _clear_cache(self):
        """Clear the frame cache"""
        self._frame_cache.clear()


class SpeedrunComparisonTool:
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

    def get_current_frame(self, video_id):
        if video_id not in self.videos:
            return 0
        player = self.videos[video_id]['player']
        return player.current_frame if player.video_capture else 0

    def get_total_frames(self, video_id):
        if video_id not in self.videos:
            return 0
        player = self.videos[video_id]['player']
        return player.total_frames if player.video_capture else 0

    def get_fps(self, video_id):
        if video_id not in self.videos:
            return 0
        player = self.videos[video_id]['player']
        return player.fps if player.video_capture else 0

    def get_video_path(self, video_id):
        if video_id not in self.videos:
            return None
        player = self.videos[video_id]['player']
        return player.video_path if player.video_capture else None

    def reset_marks(self, video_id):
        if video_id not in self.videos:
            return
        video_data = self.videos[video_id]
        video_data['start_frame'] = 0
        video_data['end_frame'] = 0
        getattr(self, f'marked_info_{video_id}').configure(text="Start: - | End: -")

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
    
    def get_marked_frames(self, video_id):
        if video_id not in self.videos:
            return 0, 0
        video_data = self.videos[video_id]
        return video_data['start_frame'], video_data['end_frame']
    
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
    
    def __init__(self, root):
        self.root = root
        self.root.title("Speedrun Comparison Tool")
        self.root.geometry("1600x1000") 
        self.set_app_icon()
        
        self.videos = {}
        self.video_counter = 0
        self.max_videos = 9
        
        try:
            import ctypes as ct
            import ctypes.wintypes as wintypes
            
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
        
        self.settings_file = ".\\app_settings.json"
        
        self.compression_settings = {
            'fps': 60,  
            'scale': 0.5,
            'codec': 'auto'
        }
        
        self.load_settings()
        
        self.gpu_available = self.check_gpu_capabilities()
        
        self.setup_gui()
        self.setup_dark_theme()
        
        self.add_video()
        self.add_video()

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
            self.settings_info.configure(text=info_text)
        except (ValueError, KeyError):
            pass

    def setup_gui(self):
        main_frame = ttk.Frame(self.root, padding="10", style="Dark.TFrame")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        settings_frame = ttk.LabelFrame(main_frame, text="Export Settings", padding="10", style="Dark.TLabelframe")
        settings_frame.pack(fill=tk.X, pady=(0, 10))
        
        settings_grid = ttk.Frame(settings_frame, style="Dark.TFrame")
        settings_grid.pack(fill=tk.X)
        
        ttk.Label(settings_grid, text="Output FPS:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.fps_var = tk.StringVar(value=str(int(self.compression_settings['fps'])))
        fps_combo = ttk.Combobox(settings_grid, textvariable=self.fps_var, values=["24", "25", "30", "60"], 
                                width=8, state="readonly")
        fps_combo.grid(row=0, column=1, padx=(0, 20), sticky=tk.W)
        fps_combo.bind('<<ComboboxSelected>>', self._update_settings)
        
        ttk.Label(settings_grid, text="Resolution:").grid(row=0, column=2, sticky=tk.W, padx=(0, 5))
        self.scale_var = tk.StringVar(value=str(self.compression_settings['scale']))
        scale_combo = ttk.Combobox(settings_grid, textvariable=self.scale_var, 
                                  values=["0.25", "0.5", "1.0"], width=8, state="readonly")
        scale_combo.grid(row=0, column=3, padx=(0, 20), sticky=tk.W)
        scale_combo.bind('<<ComboboxSelected>>', self._update_settings)
        
        ttk.Label(settings_grid, text="Codec:").grid(row=0, column=4, sticky=tk.W, padx=(0, 5))
        self.codec_var = tk.StringVar(value=self.compression_settings.get('codec', 'auto'))
        codec_combo = ttk.Combobox(settings_grid, textvariable=self.codec_var, 
                                  values=["auto", "h264", "mp4v", "xvid"], width=8, state="readonly")
        codec_combo.grid(row=0, column=5, padx=(0, 20), sticky=tk.W)
        codec_combo.bind('<<ComboboxSelected>>', self._update_settings)
        
        settings_buttons = ttk.Frame(settings_grid, style="Dark.TFrame")
        settings_buttons.grid(row=0, column=6, padx=(20, 0), sticky=tk.W)
        
        self.settings_info = ttk.Label(settings_grid, text="Half res, 30fps, auto codec",
                                      font=("Arial", 8), foreground="gray")
        self.settings_info.grid(row=1, column=0, columnspan=7, sticky=tk.W, pady=(5, 0))
        
        video_controls_frame = ttk.Frame(main_frame, style="Dark.TFrame")
        video_controls_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.add_video_btn = ttk.Button(video_controls_frame, text="+ Add Video", 
                                       command=self.add_video)
        self.add_video_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(video_controls_frame, text="Videos:", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=(20, 5))
        self.video_count_label = ttk.Label(video_controls_frame, text="0")
        self.video_count_label.pack(side=tk.LEFT)
        
        self.videos_scroll_frame = ttk.Frame(main_frame, style="Dark.TFrame")
        self.videos_scroll_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(self.videos_scroll_frame, bg="#0f0f0f", highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self.videos_scroll_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas, style="Dark.TFrame")
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        def _on_mousewheel(event):
            self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        self.canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
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
        info_text = f"Start: {start_frame} | End: {end_frame}"
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
        
        self.apply_dark_theme_to_window(progress_window)
        
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
                                        command=lambda: setattr(self, '_cancel_generation', True))
        self.gen_cancel_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.gen_pause_btn = ttk.Button(button_frame, text="Pause", 
                                       command=self._toggle_generation_pause)
        self.gen_pause_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(button_frame, text="Clear Log", 
                  command=lambda: self.gen_log_text.delete(1.0, tk.END)).pack(side=tk.LEFT)
        
        self.auto_scroll_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(button_frame, text="Auto-scroll", 
                       variable=self.auto_scroll_var).pack(side=tk.RIGHT)
        
        self.compression_settings = compression_settings
        self._cancel_generation = False
        self._pause_generation = False
        self._generation_start_time = time.time()
        
        def generate_thread():
            try:
                self._generate_video_detailed(output_path, progress_window)
            except Exception as e:
                self._log_operation(f"ERROR: {str(e)}", "error")
                self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to generate video: {str(e)}"))
            finally:
                self.root.after(0, lambda: self._finalize_generation(progress_window, output_path))
        
        thread = threading.Thread(target=generate_thread)
        thread.daemon = True
        thread.start()

    def _toggle_generation_pause(self):
        self._pause_generation = not self._pause_generation
        if self._pause_generation:
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
            self.gen_log_text.insert(tk.END, log_entry)
            self.gen_log_text.tag_add(level, f"end-{len(log_entry)}c", "end-1c")
            self.gen_log_text.tag_config(level, foreground=colors.get(level, "black"))
            
            if self.auto_scroll_var.get():
                self.gen_log_text.see(tk.END)
        
        self.root.after(0, update_log)
    
    def _update_generation_progress(self, current, total, operation="", extra_info=""):
        if total > 0:
            progress = (current / total) * 100
            elapsed = time.time() - self._generation_start_time
            
            def update_ui():
                try:
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
    
    def _generate_video_detailed(self, output_path, progress_window):
        self._log_operation("Starting video generation process", "info")
        
        settings = getattr(self, 'compression_settings', {'fps': 60, 'scale': 0.5})
        
        loaded_videos = {vid: data for vid, data in self.videos.items() if data['player'].video_capture}
        
        if len(loaded_videos) < 2:
            raise ValueError("At least 2 videos must be loaded and marked")
        
        self._log_operation("Calculating video parameters...", "info")
        
        video_durations = {}
        video_duration_frames = {}
        
        for video_id, video_data in loaded_videos.items():
            player = video_data['player']
            start_frame = video_data['start_frame']
            end_frame = video_data['end_frame']
            
            if start_frame >= end_frame:
                raise ValueError(f"Video {video_id} has invalid frame range")
            
            duration_frames = end_frame - start_frame
            duration_time = duration_frames / player.fps
            
            video_durations[video_id] = duration_time
            video_duration_frames[video_id] = duration_frames
            
            video_name = video_data['custom_name']
            self._log_operation(f"{video_name}: {duration_frames} frames ({duration_time:.2f}s)", "debug")
        
        max_duration_time = max(video_durations.values())
        
        video_ids = list(loaded_videos.keys())
        base_video_id = video_ids[0]
        base_duration = video_durations[base_video_id]
        base_name = loaded_videos[base_video_id]['custom_name']
        
        for video_id in video_ids[1:]:
            video_name = loaded_videos[video_id]['custom_name']
            time_diff = video_durations[video_id] - base_duration
            self._log_operation(f"{video_name} vs {base_name}: {time_diff:.3f}s difference", "debug")
        
        max_duration = max_duration_time + 2.0
        output_fps = float(settings['fps'])  
        total_output_frames = int(max_duration * output_fps)
        
        self._log_operation(f"Output duration: {max_duration:.2f}s at {output_fps} fps", "info")
        self._log_operation(f"Total output frames: {total_output_frames}", "info")
        self._log_operation(f"Compression: {settings['scale']}x scale", "info")
        
        self._log_operation("Reading test frames for dimension calculation...", "info")
        
        video_dimensions = {}
        for video_id, video_data in loaded_videos.items():
            player = video_data['player']
            start_frame = video_data['start_frame']
            test_frame = player.get_frame_fast(start_frame)
            
            if test_frame is None:
                raise ValueError(f"Could not read test frame from video {video_id}")
            
            h, w = test_frame.shape[:2]
            scale = settings['scale']
            w_scaled = int(w * scale)
            h_scaled = int(h * scale)
            
            video_dimensions[video_id] = {
                'original': (w, h),
                'scaled': (w_scaled, h_scaled)
            }
            
            video_name = video_data['custom_name']
            self._log_operation(f"{video_name} dimensions: {w}x{h} -> {w_scaled}x{h_scaled}", "debug")
        
        import math
        
        num_videos = len(loaded_videos)
        spacing = int(10 * settings['scale'])
        
        cols = math.ceil(math.sqrt(num_videos))
        rows = math.ceil(num_videos / cols)
        
        max_width = max(dims['scaled'][0] for dims in video_dimensions.values())
        max_height = max(dims['scaled'][1] for dims in video_dimensions.values())
        
        output_width = cols * max_width + (cols - 1) * spacing
        output_height = rows * max_height + (rows - 1) * spacing + int(120 * settings['scale'])
        
        self._log_operation(f"Grid layout: {rows} rows x {cols} columns", "info")
        self._log_operation(f"Output dimensions: {output_width}x{output_height}", "info")
        
        self._log_operation("Initializing video writer...", "info")
        
        preferred_codec = self.compression_settings.get('codec', 'auto')
        
        if preferred_codec == 'auto':
            codecs_to_try = [
                ('h264', 'H.264 (best web compatibility)'),
                ('avc1', 'AVC1 (H.264 variant)'),
                ('mp4v', 'MPEG-4 (good compatibility)'),
                ('XVID', 'Xvid (fallback)'),
                ('MJPG', 'Motion JPEG (universal fallback)')
            ]
        else:
            codec_map = {
                'h264': [('h264', 'H.264 (user selected)'), ('avc1', 'AVC1 fallback')],
                'mp4v': [('mp4v', 'MPEG-4 (user selected)')],
                'xvid': [('XVID', 'Xvid (user selected)')]
            }
            
            codecs_to_try = codec_map.get(preferred_codec, [])
            codecs_to_try.extend([
                ('h264', 'H.264 (fallback)'),
                ('mp4v', 'MPEG-4 (fallback)'),
                ('XVID', 'Xvid (fallback)'),
                ('MJPG', 'Motion JPEG (universal fallback)')
            ])
            seen = set()
            codecs_to_try = [(c, d) for c, d in codecs_to_try if not (c in seen or seen.add(c))]
        
        out = None
        used_codec = None
        
        for codec, description in codecs_to_try:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                out = cv2.VideoWriter(output_path, fourcc, output_fps, (output_width, output_height))
                
                if out and out.isOpened():
                    used_codec = codec
                    self._log_operation(f"Using codec: {codec} ({description})", "success")
                    break
                else:
                    if out:
                        out.release()
                    self._log_operation(f"Codec {codec} failed, trying next...", "warning")
            except Exception as e:
                self._log_operation(f"Codec {codec} error: {str(e)}", "warning")
                if out:
                    out.release()
                    out = None
        
        if not out or not out.isOpened():
            raise ValueError("Could not initialize video writer with any codec. Please check output path and try again.")
        
        self._log_operation(f"Video writer initialized successfully with {used_codec} codec", "success")
        
        self.gen_progress_bar['maximum'] = total_output_frames
        self._update_generation_progress(0, total_output_frames, "Setting up parallel processing")
        
        self._log_operation("Starting parallel frame processing with compositing...", "info")
        
        frame_queues = {video_id: queue.Queue(maxsize=100) for video_id in loaded_videos.keys()}
        composition_queue = queue.Queue(maxsize=50)
        
        processing_state = {
            'cancel': False,
            'frames_composed': 0,
            'frames_written': 0,
            'composition_complete': False
        }
        
        for video_id in loaded_videos.keys():
            processing_state[f'frames_read_{video_id}'] = 0
            processing_state[f'reading_complete_{video_id}'] = False
        
        def read_video_frames(video_id, video_path, start_frame, max_frames, frame_queue, target_width, target_height):
            try:
                cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
                if not cap.isOpened():
                    cap = cv2.VideoCapture(video_path)
                
                if not cap.isOpened():
                    self._log_operation(f"Failed to open video {video_id} for reading", "error")
                    return
                
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                
                frames_read = 0
                batch_size = 10
                
                while frames_read < max_frames and not self._cancel_generation and not processing_state['cancel']:
                    batch = []
                    
                    for _ in range(min(batch_size, max_frames - frames_read)):
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        frame_resized = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
                        batch.append((frames_read, frame_resized))
                        frames_read += 1
                    
                    for frame_data in batch:
                        frame_queue.put(frame_data)
                    
                    processing_state[f'frames_read_{video_id}'] = frames_read
                
                cap.release()
                frame_queue.put(None)
                processing_state[f'reading_complete_{video_id}'] = True
                self._log_operation(f"Completed reading {frames_read} frames from video {video_id}", "success")
                
            except Exception as e:
                self._log_operation(f"Error reading video {video_id}: {str(e)}", "error")
                frame_queue.put(None)
        
        def compose_frames():
            try:
                video_frame_caches = {video_id: {} for video_id in loaded_videos.keys()}
                cache_limit = 500
                
                for frame_idx in range(total_output_frames):
                    if self._cancel_generation or processing_state['cancel']:
                        break
                    
                    current_time = frame_idx / output_fps
                    
                    video_states = {}
                    for video_id, video_data in loaded_videos.items():
                        player = video_data['player']
                        duration_time = video_durations[video_id]
                        
                        video_active = current_time <= duration_time
                        relative_frame = int(current_time * player.fps) if video_active else -1
                        
                        video_states[video_id] = {
                            'active': video_active,
                            'relative_frame': relative_frame,
                            'duration': duration_time
                        }
                    
                    max_wait_time = 5.0  
                    wait_start = time.time()
                    
                    waiting_for_frames = True
                    while waiting_for_frames and not self._cancel_generation:
                        waiting_for_frames = False
                        
                        for video_id, state in video_states.items():
                            if (state['active'] and state['relative_frame'] >= 0 and 
                                state['relative_frame'] not in video_frame_caches[video_id] and 
                                not processing_state[f'reading_complete_{video_id}']):
                                waiting_for_frames = True
                                break
                        
                        if not waiting_for_frames:
                            break
                        
                        for video_id in loaded_videos.keys():
                            self._update_frame_cache(frame_queues[video_id], video_frame_caches[video_id], 
                                                   video_states[video_id]['relative_frame'], cache_limit)
                        
                        if time.time() - wait_start > max_wait_time:
                            self._log_operation(f"Timeout waiting for frames at output frame {frame_idx}", "warning")
                            break
                        
                        time.sleep(0.001)
                    
                    for video_id in loaded_videos.keys():
                        self._update_frame_cache(frame_queues[video_id], video_frame_caches[video_id], 
                                               video_states[video_id]['relative_frame'], cache_limit)
                    
                    output_frame = np.zeros((output_height, output_width, 3), dtype=np.uint8)
                    
                    scale = settings['scale']
                    font_scale_base = 1.2 * scale  
                    font_thickness = max(1, int(2 * scale))
                    time_font_scale = 2.0 * scale 
                    time_thickness = max(2, int(3 * scale))
                    
                    for i, (video_id, video_data) in enumerate(loaded_videos.items()):
                        state = video_states[video_id]
                        dimensions = video_dimensions[video_id]
                        w_scaled, h_scaled = dimensions['scaled']
                        
                        grid_row = i // cols
                        grid_col = i % cols
                        
                        x_offset = grid_col * (max_width + spacing) + (max_width - w_scaled) // 2
                        y_offset = int(60 * scale) + grid_row * (max_height + spacing) + (max_height - h_scaled) // 2
                        
                        video_name = video_data['custom_name']
                        
                        name_x = grid_col * (max_width + spacing) + int(10 * scale)
                        name_y = int(30 * scale) + grid_row * (max_height + spacing + int(30 * scale))
                        
                        cv2.putText(
                            output_frame, video_name,
                            (name_x, name_y),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale_base, (255, 255, 255), font_thickness)
                        
                        if state['active'] and state['relative_frame'] >= 0:
                            if state['relative_frame'] in video_frame_caches[video_id]:
                                frame = video_frame_caches[video_id][state['relative_frame']]
                                output_frame[y_offset:y_offset + h_scaled, x_offset:x_offset + w_scaled] = frame
                        
                        if not state['active'] and current_time > state['duration']:
                            time_text = f"{state['duration']:.3f}s"
                            time_size = cv2.getTextSize(time_text, cv2.FONT_HERSHEY_SIMPLEX, time_font_scale, 1)[0] 
                            time_x = x_offset + (w_scaled - time_size[0]) // 2
                            time_y = y_offset + h_scaled // 2
                            cv2.putText(output_frame, time_text, (time_x, time_y),
                                        cv2.FONT_HERSHEY_SIMPLEX, time_font_scale, (255, 255, 255), 1)

                            fastest_duration = min(video_durations.values())
                            if state['duration'] != fastest_duration:
                                time_diff = state['duration'] - fastest_duration
                                diff_text = f"+{time_diff:.3f}s"
                                diff_font_scale = 1.0 * scale
                                diff_thickness = 1  
                                diff_size = cv2.getTextSize(diff_text, cv2.FONT_HERSHEY_SIMPLEX, diff_font_scale, diff_thickness)[0]
                                
                                diff_x = x_offset + (w_scaled - diff_size[0]) // 2
                                diff_y = y_offset + h_scaled // 2 + int(40 * scale)
                                cv2.putText(output_frame, diff_text, (diff_x, diff_y),
                                            cv2.FONT_HERSHEY_SIMPLEX, diff_font_scale, (0, 0, 255), diff_thickness)  
                    
                    composition_queue.put((frame_idx, output_frame))
                    processing_state['frames_composed'] = frame_idx + 1
                
                composition_queue.put(None) 
                processing_state['composition_complete'] = True
                self._log_operation("Frame composition thread completed", "success")
                
            except Exception as e:
                self._log_operation(f"Error in composition: {str(e)}", "error")
                composition_queue.put(None)
        
        reader_threads = []
        for video_id, video_data in loaded_videos.items():
            player = video_data['player']
            start_frame = video_data['start_frame']
            duration_frames = video_duration_frames[video_id]
            w_scaled, h_scaled = video_dimensions[video_id]['scaled']
            
            thread = threading.Thread(target=read_video_frames, args=(
                video_id, player.video_path, start_frame, duration_frames, 
                frame_queues[video_id], w_scaled, h_scaled))
            reader_threads.append(thread)
        
        composer_thread = threading.Thread(target=compose_frames)
        
        for thread in reader_threads + [composer_thread]:
            thread.daemon = True
            thread.start()
        
        self._log_operation("Starting video writer loop...", "info")
        frame_composition_start = time.time()
        frames_written = 0
        
        while frames_written < total_output_frames and not self._cancel_generation:
            while self._pause_generation and not self._cancel_generation:
                time.sleep(0.1)
            
            try:
                frame_data = composition_queue.get(timeout=10.0)
                if frame_data is None:
                    break
                
                frame_idx, output_frame = frame_data
                out.write(output_frame)
                frames_written += 1
                processing_state['frames_written'] = frames_written
                
                if frames_written % 15 == 0:  
                    read_stats = []
                    for video_id in loaded_videos.keys():
                        read_count = processing_state[f'frames_read_{video_id}']
                        read_stats.append(f"{video_id}:{read_count}")
                    cache_info = "R[" + ",".join(read_stats) + "]"
                    
                    self._update_generation_progress(frames_written, total_output_frames, 
                                                   f"Writing frame {frames_written}/{total_output_frames}", 
                                                   cache_info)
                    
                    def update_memory():
                        memory_text = f"Read: {cache_info} | Composed: {processing_state['frames_composed']}"
                        self.gen_memory_label.configure(text=f"Memory: {memory_text}")
                    self.root.after(0, update_memory)
                
                if frames_written % 300 == 0: 
                    elapsed = time.time() - frame_composition_start
                    write_fps = frames_written / elapsed if elapsed > 0 else 0
                    self._log_operation(f"Written {frames_written} frames at {write_fps:.1f} fps", "debug")
                    
            except queue.Empty:
                if processing_state['composition_complete']:
                    self._log_operation("Composition complete but no more frames in queue", "debug")
                    break
                else:
                    self._log_operation(f"Waiting for frame {frames_written}, composition at {processing_state['frames_composed']}", "debug")
                continue
            except Exception as e:
                self._log_operation(f"Error writing frame: {str(e)}", "error")
                break
        
        processing_state['cancel'] = True
        self._log_operation("Waiting for threads to complete...", "info")
        
        for thread in reader_threads + [composer_thread]:
            thread.join(timeout=3.0)
        
        out.release()
        
        if not self._cancel_generation:
            composition_time = time.time() - frame_composition_start
            avg_fps = frames_written / composition_time if composition_time > 0 else 0
            self._log_operation(f"Video generation completed in {composition_time:.1f}s", "success")
            self._log_operation(f"Average processing speed: {avg_fps:.1f} fps", "success")
            self._update_generation_progress(total_output_frames, total_output_frames, "Generation Complete!")
        else:
            self._log_operation("Generation cancelled - cleaning up...", "warning")
            if os.path.exists(output_path):
                os.remove(output_path)
                self._log_operation("Cancelled file removed", "info")
    
    def _update_frame_cache(self, frame_queue, cache_dict, needed_frame, cache_limit):
        try:
            while not frame_queue.empty():
                frame_data = frame_queue.get_nowait()
                if frame_data is None:
                    break
                frame_idx, frame = frame_data
                cache_dict[frame_idx] = frame
                

                if len(cache_dict) > cache_limit:
                    if needed_frame >= 0:
                        frames_to_remove = [idx for idx in cache_dict.keys() 
                                          if idx < needed_frame - 50]
                        for idx in frames_to_remove[:len(frames_to_remove)//2]:
                            del cache_dict[idx]
        except queue.Empty:
            pass
    
    def _finalize_generation(self, progress_window, output_path):
        self.gen_cancel_btn.configure(text="Close")
        self.gen_pause_btn.configure(state="disabled")
        
        if not self._cancel_generation and os.path.exists(output_path):
            messagebox.showinfo("Success", 
                f"Comparison video generated successfully!\n\n"
                f"Saved to: {output_path}\n\n"
                )
    
    def on_closing(self):
        self.save_settings()
        
        if hasattr(self, '_cancel_generation'):
            self._cancel_generation = True
        
        for video_id, video_data in self.videos.items():
            if 'player' in video_data and video_data['player']:
                video_data['player'].close()
        
        self.root.destroy()

    def setup_dark_theme(self):
        """Configure dark theme for the application"""
        style = ttk.Style()
        
        dark_bg = "#0f0f0f"
        dark_fg = "#e0e0e0"
        dark_select_bg = "#2a2a2a"
        dark_entry_bg = "#1a1a1a"
        dark_button_bg = "#2a2a2a"
        dark_frame_bg = "#0f0f0f"
        dark_text_bg = "#1a1a1a"
        
        style.theme_use('clam')
        
        style.configure('TFrame', background=dark_bg)
        
       
        try:
            style.element_create('Dark.Labelframe.border', 'from', 'default')
            style.layout('Dark.TLabelframe', [
                ('Dark.Labelframe.border', {'sticky': 'nswe', 'children': [
                    ('Labelframe.padding', {'sticky': 'nswe', 'children': [
                        ('Labelframe.label', {'side': 'left', 'sticky': ''}),
                    ]})
                ]})
            ])
            
            style.configure('Dark.TLabelframe', 
                           background=dark_bg, 
                           foreground=dark_fg,
                           bordercolor=dark_bg,
                           darkcolor=dark_bg,
                           lightcolor=dark_bg,
                           focuscolor=dark_bg,
                           borderwidth=0,
                           relief='flat')
            
            style.map('Dark.TLabelframe',
                     background=[('', dark_bg)],
                     bordercolor=[('', dark_bg)],
                     lightcolor=[('', dark_bg)],
                     darkcolor=[('', dark_bg)],
                     focuscolor=[('', dark_bg)])
                     
        except Exception as e:
            print(f"Advanced labelframe styling failed: {e}")
            style.configure('Dark.TLabelframe', 
                           background=dark_bg, 
                           foreground=dark_fg,
                           borderwidth=0,
                           relief='flat')
        
        style.configure('Dark.TLabelframe.Label', 
                       background=dark_bg, 
                       foreground=dark_fg,
                       font=('TkDefaultFont', 9, 'bold'))
        
        style.configure('Dark.TFrame', 
                       background=dark_bg,
                       borderwidth=0,
                       relief='flat')
        
        style.configure('TLabelFrame', 
                       background=dark_bg, 
                       foreground=dark_fg, 
                       borderwidth=0, 
                       relief='flat',
                       lightcolor=dark_bg,
                       darkcolor=dark_bg)
        style.configure('TLabelFrame.Label', background=dark_bg, foreground=dark_fg)
        
        style.configure('TLabel', background=dark_bg, foreground=dark_fg)
        
        style.configure('Dark.TLabel', 
                       background=dark_bg, 
                       foreground=dark_fg,
                       font=('TkDefaultFont', 10, 'normal'))
        
        style.configure('TButton', 
                       background=dark_button_bg, 
                       foreground=dark_fg,
                       borderwidth=1,
                       focuscolor='none')
        style.map('TButton',
                 background=[('active', '#565656'), ('pressed', '#6a6a6a')])
        
        style.configure('TEntry', 
                       background=dark_entry_bg, 
                       foreground=dark_fg,
                       borderwidth=1,
                       insertcolor=dark_fg,
                       fieldbackground=dark_entry_bg)
        style.map('TEntry',
                 background=[('focus', dark_entry_bg), ('active', dark_entry_bg)],
                 fieldbackground=[('focus', dark_entry_bg), ('active', dark_entry_bg)])
        
        style.configure('TCombobox', 
                       background=dark_entry_bg, 
                       foreground=dark_fg,
                       borderwidth=1,
                       arrowcolor=dark_fg,
                       fieldbackground=dark_entry_bg,
                       selectbackground=dark_select_bg,
                       insertcolor=dark_fg)
        style.map('TCombobox',
                 background=[('focus', dark_entry_bg), ('active', dark_entry_bg)],
                 fieldbackground=[('focus', dark_entry_bg), ('active', dark_entry_bg)],
                 selectbackground=[('focus', dark_select_bg)],
                 arrowcolor=[('active', dark_fg)])
        
        style.configure('TScale',
                       background=dark_bg,
                       troughcolor=dark_entry_bg,
                       borderwidth=1,
                       darkcolor=dark_button_bg,
                       lightcolor=dark_button_bg)
        
        style.configure('TProgressbar',
                       background='#4a9eff',
                       troughcolor=dark_entry_bg,
                       borderwidth=1,
                       lightcolor='#4a9eff',
                       darkcolor='#4a9eff')
        
        style.configure('TCheckbutton',
                       background=dark_bg,
                       foreground=dark_fg,
                       focuscolor='none')
        style.map('TCheckbutton',
                 background=[('active', dark_bg)])
        
        style.configure('TScrollbar',
                       background=dark_button_bg,
                       darkcolor=dark_entry_bg,
                       lightcolor=dark_button_bg,
                       troughcolor=dark_bg,
                       bordercolor=dark_entry_bg,
                       arrowcolor=dark_fg)
        
        self.root.configure(bg=dark_bg)
        
        self.root.option_add('*Text.background', dark_text_bg)
        self.root.option_add('*Text.foreground', dark_fg)
        self.root.option_add('*Text.insertBackground', dark_fg)
        self.root.option_add('*Text.selectBackground', dark_select_bg)
        self.root.option_add('*Text.selectForeground', dark_fg)
        
        self.root.option_add('*Listbox.background', dark_entry_bg)
        self.root.option_add('*Listbox.foreground', dark_fg)
        self.root.option_add('*Listbox.selectBackground', dark_select_bg)
        self.root.option_add('*Listbox.selectForeground', dark_fg)
        
        self.root.option_add('*Menu.background', dark_bg)
        self.root.option_add('*Menu.foreground', dark_fg)
        self.root.option_add('*Menu.selectColor', dark_select_bg)
        
        self.root.option_add('*TCombobox*Listbox.background', dark_entry_bg)
        self.root.option_add('*TCombobox*Listbox.foreground', dark_fg)
        self.root.option_add('*TCombobox*Listbox.selectBackground', dark_select_bg)
        self.root.option_add('*TCombobox*Listbox.selectForeground', dark_fg)
        
        self.root.option_add('*background', dark_bg)
        self.root.option_add('*foreground', dark_fg)
        self.root.option_add('*selectBackground', dark_select_bg)
        self.root.option_add('*selectForeground', dark_fg)
        self.root.option_add('*highlightBackground', dark_bg)
        self.root.option_add('*highlightColor', dark_select_bg)
        self.root.option_add('*troughColor', dark_entry_bg)
        self.root.option_add('*activeBackground', dark_select_bg)
        self.root.option_add('*activeForeground', dark_fg)
        
    def apply_dark_theme_to_window(self, window):
        """Apply dark theme to a specific window or dialog"""
        dark_bg = "#0f0f0f"
        dark_fg = "#e0e0e0"
        
        try:
            window.configure(bg=dark_bg)
            
            def configure_widget(widget):
                widget_class = widget.winfo_class()
                if widget_class == 'Text':
                    widget.configure(bg="#1a1a1a", fg=dark_fg, 
                                   insertbackground=dark_fg,
                                   selectbackground="#2a2a2a",
                                   selectforeground=dark_fg)
                elif widget_class in ['Frame', 'Toplevel']:
                    widget.configure(bg=dark_bg)
                elif widget_class == 'Label':
                    widget.configure(bg=dark_bg, fg=dark_fg)
                
                for child in widget.winfo_children():
                    configure_widget(child)
            
            configure_widget(window)
        except:
            pass  
    
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
            'custom_name': f'Video {video_id}'
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
        videos_per_row = min(3, len(self.videos)) if len(self.videos) > 0 else 1
        
        for i, video_id in enumerate(self.videos.keys()):
            if hasattr(self, f'video_panel_{video_id}'):
                panel = getattr(self, f'video_panel_{video_id}')
                row = i // videos_per_row
                col = i % videos_per_row
                panel.grid(row=row, column=col, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        for col in range(videos_per_row):
            self.scrollable_frame.columnconfigure(col, weight=1)

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
        
        time_info = ttk.Label(controls, text="Time: 0.00s", font=("Courier", 9))
        time_info.pack(side=tk.LEFT, padx=(20, 0))
        setattr(self, f'time_info_{video_id}', time_info)
        
        mark_frame = ttk.Frame(panel, style="Dark.TFrame")
        mark_frame.pack(fill=tk.X)
        
        ttk.Button(mark_frame, text="Mark Start", 
                  command=lambda: self.mark_frame(video_id, 'start')).pack(side=tk.LEFT, padx=2)
        ttk.Button(mark_frame, text="Mark End", 
                  command=lambda: self.mark_frame(video_id, 'end')).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(mark_frame, text="Jump Start", width=12,
                  command=lambda: self.jump_to_mark(video_id, 'start')).pack(side=tk.LEFT, padx=2)
        ttk.Button(mark_frame, text="Jump End", width=12,
                  command=lambda: self.jump_to_mark(video_id, 'end')).pack(side=tk.LEFT, padx=2)
        
        marked_info = ttk.Label(mark_frame, text="Start: - | End: -", font=("Courier", 8))
        marked_info.pack(side=tk.RIGHT)
        setattr(self, f'marked_info_{video_id}', marked_info)

if __name__ == "__main__":
    root = tk.Tk()
    app = SpeedrunComparisonTool(root)
    root.mainloop()




