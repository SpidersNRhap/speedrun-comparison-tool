import cv2
import numpy as np
import threading
import time
import queue
import os
import math

class VideoGenerator:
    def __init__(self, log_callback, progress_callback):
        self._log_operation = log_callback
        self._update_generation_progress = progress_callback
        self._cancel_generation = False
        self._pause_generation = False

    def set_cancel_flag(self, value):
        self._cancel_generation = value

    def set_pause_flag(self, value):
        self._pause_generation = value

    def generate_comparison_video(self, output_path, loaded_videos, compression_settings):
        self._log_operation("Starting video generation process", "info")

        settings = compression_settings

        if len(loaded_videos) < 2:
            raise ValueError("At least 2 videos must be loaded and marked")

        audio_enabled_videos = [vid for vid, data in loaded_videos.items() 
                              if data.get('audio_enabled', False)]

        if audio_enabled_videos:
            self._log_operation(f"Audio enabled for videos: {audio_enabled_videos}", "info")
        else:
            self._log_operation("No audio will be included in output", "info")

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

        max_duration = max_duration_time + 3.0  
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

        num_videos = len(loaded_videos)
        spacing = int(20 * settings['scale'])  

        cols = math.ceil(math.sqrt(num_videos))
        rows = math.ceil(num_videos / cols)

        max_width = max(dims['scaled'][0] for dims in video_dimensions.values())
        max_height = max(dims['scaled'][1] for dims in video_dimensions.values())

        output_width = cols * max_width + (cols - 1) * spacing
        output_height = rows * max_height + (rows - 1) * spacing + int(100 * settings['scale'])  

        self._log_operation(f"Grid layout: {rows} rows x {cols} columns", "info")
        self._log_operation(f"Output dimensions: {output_width}x{output_height}", "info")

        self._log_operation("Initializing video writer...", "info")

        out = self._initialize_video_writer(output_path, output_fps, output_width, output_height, settings)

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

        reader_threads = []
        for video_id, video_data in loaded_videos.items():
            player = video_data['player']
            start_frame = video_data['start_frame']
            duration_frames = video_duration_frames[video_id]
            w_scaled, h_scaled = video_dimensions[video_id]['scaled']

            thread = threading.Thread(target=self._read_video_frames, args=(
                video_id, player.video_path, start_frame, duration_frames, 
                frame_queues[video_id], w_scaled, h_scaled, processing_state))
            reader_threads.append(thread)

        composer_thread = threading.Thread(target=self._compose_frames, args=(
            loaded_videos, video_durations, video_dimensions, settings, 
            total_output_frames, output_fps, cols, rows, max_width, max_height, 
            spacing, output_width, output_height, frame_queues, composition_queue, processing_state))

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

        if audio_enabled_videos and not self._cancel_generation:
            self._add_multiple_audio_tracks(output_path, loaded_videos, audio_enabled_videos, 
                                          video_durations, max_duration)

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

    def _initialize_video_writer(self, output_path, output_fps, output_width, output_height, settings):
        preferred_codec = settings.get('codec', 'auto')

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
        return out

    def _read_video_frames(self, video_id, video_path, start_frame, max_frames, frame_queue, target_width, target_height, processing_state):
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

    def _compose_frames(self, loaded_videos, video_durations, video_dimensions, settings, 
                    total_output_frames, output_fps, cols, rows, max_width, max_height, 
                    spacing, output_width, output_height, frame_queues, composition_queue, processing_state):
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
                    y_offset = int(40 * scale) + grid_row * (max_height + spacing) + (max_height - h_scaled) // 2  

                    video_name = video_data['custom_name']

                    name_x = grid_col * (max_width + spacing) + int(10 * scale)
                    name_y = int(30 * scale) + grid_row * (max_height + spacing + int(25 * scale))  

                    cv2.putText(
                        output_frame, video_name,
                        (name_x, name_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale_base, (255, 255, 255), font_thickness)

                    if state['active'] and state['relative_frame'] >= 0:
                        if state['relative_frame'] in video_frame_caches[video_id]:
                            frame = video_frame_caches[video_id][state['relative_frame']]
                            output_frame[y_offset:y_offset + h_scaled, x_offset:x_offset + w_scaled] = frame

                    if not state['active'] and current_time > state['duration']:
                        time_text = f"{state['duration']:6.3f}s"  
                        text_size = cv2.getTextSize(time_text, cv2.FONT_HERSHEY_SIMPLEX, time_font_scale, time_thickness)[0]
                        time_x = x_offset + (w_scaled - text_size[0]) // 2
                        time_y = y_offset + h_scaled // 2

                        cv2.putText(output_frame, time_text, (time_x, time_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, time_font_scale, (255, 255, 255), time_thickness)

                        fastest_duration = min(video_durations.values())
                        if state['duration'] != fastest_duration:
                            time_diff = state['duration'] - fastest_duration
                            diff_text = f"+{time_diff:5.3f}s"  
                            diff_font_scale = 1.0 * scale
                            diff_thickness = max(1, int(2 * scale))
                            diff_size = cv2.getTextSize(diff_text, cv2.FONT_HERSHEY_SIMPLEX, diff_font_scale, diff_thickness)[0]

                            diff_x = x_offset + (w_scaled - diff_size[0]) // 2
                            diff_y = y_offset + h_scaled // 2 + int(40 * scale)
                            cv2.putText(output_frame, diff_text, (diff_x, diff_y),
                                        cv2.FONT_HERSHEY_SIMPLEX, diff_font_scale, (0, 0, 255), diff_thickness)

                timer_text = f"{current_time:.2f}s"
                timer_font_scale = 2.5 * scale
                timer_thickness = max(2, int(4 * scale))
                timer_size = cv2.getTextSize(timer_text, cv2.FONT_HERSHEY_SIMPLEX, timer_font_scale, timer_thickness)[0]
                timer_x = (output_width - timer_size[0]) // 2
                timer_y = output_height - int(20 * scale)  

                outline_thickness = max(3, int(6 * scale))
                cv2.putText(output_frame, timer_text,
                        (timer_x, timer_y),
                        cv2.FONT_HERSHEY_SIMPLEX, timer_font_scale, (0, 0, 0), outline_thickness)  

                cv2.putText(output_frame, timer_text,
                        (timer_x, timer_y),
                        cv2.FONT_HERSHEY_SIMPLEX, timer_font_scale, (255, 255, 255), timer_thickness)  

                composition_queue.put((frame_idx, output_frame))
                processing_state['frames_composed'] = frame_idx + 1

            composition_queue.put(None) 
            processing_state['composition_complete'] = True
            self._log_operation("Frame composition thread completed", "success")

        except Exception as e:
            self._log_operation(f"Error in composition: {str(e)}", "error")
            composition_queue.put(None)

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

    def _add_multiple_audio_tracks(self, output_path, all_videos, audio_video_ids, video_durations, max_duration):
        """Add audio tracks from multiple videos to the output"""
        try:
            import subprocess

            self._log_operation("Adding audio tracks...", "info")

            temp_output = output_path.replace('.mp4', '_temp.mp4')
            os.rename(output_path, temp_output)

            if len(audio_video_ids) == 1:

                video_id = audio_video_ids[0]
                video_data = all_videos[video_id]
                self._add_single_audio_track(temp_output, output_path, video_data, max_duration)
            else:

                self._mix_multiple_audio_tracks(temp_output, output_path, all_videos, 
                                              audio_video_ids, max_duration)

            if os.path.exists(output_path):
                os.remove(temp_output)
                self._log_operation("Audio tracks added successfully", "success")
            else:
                os.rename(temp_output, output_path)  
                self._log_operation("Failed to add audio tracks", "warning")

        except Exception as e:
            self._log_operation(f"Error adding audio: {str(e)}", "warning")
            if os.path.exists(temp_output):
                os.rename(temp_output, output_path)  

    def _add_single_audio_track(self, temp_output, output_path, video_data, max_duration):
        """Add a single audio track"""
        import subprocess

        player = video_data['player']
        start_frame = video_data['start_frame']
        start_time = start_frame / player.fps

        cmd = [
            'ffmpeg', '-y',
            '-i', temp_output,  
            '-i', player.video_path,  
            '-ss', str(start_time),  
            '-t', str(max_duration),  
            '-c:v', 'copy',  
            '-c:a', 'aac',  
            '-map', '0:v:0',  
            '-map', '1:a:0',  
            '-shortest',  
            output_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"FFmpeg error: {result.stderr}")

    def _mix_multiple_audio_tracks(self, temp_output, output_path, all_videos, audio_video_ids, max_duration):
        """Mix multiple audio tracks together"""
        import subprocess
        import tempfile

        temp_audio_files = []

        try:
            for i, video_id in enumerate(audio_video_ids):
                video_data = all_videos[video_id]
                player = video_data['player']
                start_frame = video_data['start_frame']
                start_time = start_frame / player.fps

                temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=f'_audio_{i}.wav')
                temp_audio_files.append(temp_audio.name)
                temp_audio.close()

                cmd = [
                    'ffmpeg', '-y',
                    '-i', player.video_path,
                    '-ss', str(start_time),
                    '-t', str(max_duration),
                    '-vn',  
                    '-acodec', 'pcm_s16le',  
                    '-ar', '44100',  
                    '-ac', '2',  
                    temp_audio.name
                ]

                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    self._log_operation(f"Failed to extract audio from video {video_id}: {result.stderr}", "warning")
                    continue

            if len(temp_audio_files) == 0:
                raise Exception("No audio tracks could be extracted")

            if len(temp_audio_files) == 1:
                mixed_audio = temp_audio_files[0]
            else:
                mixed_audio = tempfile.NamedTemporaryFile(delete=False, suffix='_mixed.wav')
                mixed_audio.close()

                inputs = []
                filter_inputs = []
                for i, audio_file in enumerate(temp_audio_files):
                    inputs.extend(['-i', audio_file])
                    filter_inputs.append(f'[{i}:a]')

                filter_complex = f"{''.join(filter_inputs)}amix=inputs={len(temp_audio_files)}:duration=longest[a]"

                cmd = ['ffmpeg', '-y'] + inputs + [
                    '-filter_complex', filter_complex,
                    '-map', '[a]',
                    '-acodec', 'pcm_s16le',
                    mixed_audio.name
                ]

                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    raise Exception(f"Audio mixing failed: {result.stderr}")

                mixed_audio = mixed_audio.name

            cmd = [
                'ffmpeg', '-y',
                '-i', temp_output,  
                '-i', mixed_audio,  
                '-c:v', 'copy',  
                '-c:a', 'aac',  
                '-map', '0:v:0',  
                '-map', '1:a:0',  
                '-shortest',
                output_path
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception(f"Final audio mixing failed: {result.stderr}")

        finally:

            for temp_file in temp_audio_files:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            if 'mixed_audio' in locals() and mixed_audio != temp_audio_files[0] and os.path.exists(mixed_audio):
                os.remove(mixed_audio)
                os.remove(mixed_audio)