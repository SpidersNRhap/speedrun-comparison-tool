import cv2
import numpy as np
import threading
import time
import queue
from tkinter import messagebox

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