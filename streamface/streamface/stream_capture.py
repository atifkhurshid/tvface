import av
import time
import streamlink
import traceback

from datetime import datetime
from pathlib import Path
from PIL import Image

from .utils import Logging
from .utils import DelayedKeyboardInterrupt

from .keyframe_extraction import KeyframeExtraction


class StreamCapture(object):
    """Capture live video stream and store selected frames."""

    def __init__(self,
            name,
            url,
            output_dir,
            batch_size=50,
            empty_threshold=0.95,
            blur_threshold=50,
            similarity_threshold=0.9,
            reconnect_interval=500,
            log_interval=100
        ):
                 
        self.name = name
        self.url = url
        self.batch_size = batch_size
        self.reconnect_interval = reconnect_interval
        self.log_interval = log_interval

        self.paths = self.getpaths(output_dir)
        self.logger = self.getlogger()

        self.keyframes = KeyframeExtraction(
            empty_threshold, blur_threshold, similarity_threshold)

        self.container = None
        self.frames = {}


    def call(self):
        """Process live video in an infinite loop
        
        In each loop:
            - Get next frame from video stream
            - Select key frame and save

        Use KeyboardInterrupt (Ctrl + C) to end loop.
        """
        try:
            self.logger.log('Initializing Stream Capture', 'INFO')

            n = -1
            stream = self.getstream()
            
            while True:
                n += 1
                try:
                    # =========================================================== #

                    frame = next(stream)
    
                    self.capture(frame, n)

                    if (n + 1) % self.reconnect_interval == 0:
                        stream = self.getstream()
                        self.logger.flush()
                    
                    # =========================================================== #
                    
                    if n  % self.log_interval == 0:
                        msg = f'Processing - Iteration {n}'
                        self.logger.log(msg, 'INFO')
                        
                except KeyboardInterrupt:
                    msg = 'Close instruction received. Wrapping up final iteration'
                    self.logger.log(msg, 'INFO')
                    break

                except Exception:
                    msg = f'Exception - Iteration {n}'
                    self.logger.log(msg, 'ERR')
                    self.logger.log(traceback.format_exc(), 'ERR')
                    
                    stream = self.getstream()
                    continue
        finally:
            self.close()

    # =========================================================== #

    def capture(self, frame, n):
        frame = self.keyframes.getkeyframe(frame)
        if frame is not None:
            self.addframe(frame)

        if n % self.batch_size == 0:
            self.saveframes()

    # =========================================================== #


    def getstream(self):
        """Get frame generator from video stream

        Keep trying every 60s:
            - Use streamlink to get best url.
            - Use PyAV to extract keyframes
            - Return frames in a generator fn
        """
        def frame_generator(container, stream):
            corrupted = 0
            for frame in container.decode(stream):
                if not frame.is_corrupt:
                    corrupted = 0
                    yield frame.to_ndarray(format='rgb24')
                else:
                    self.logger.log('Corrupt Frame', 'ERR')
                    corrupted += 1
                    if corrupted >= 2:
                        raise StopIteration

        msg = f'Getting video stream from {self.url}'
        self.logger.log(msg, 'INFO')

        while True:
            try:
                # Manual garbage collection because of a PyAV caveat
                if self.container is not None:
                    self.container.close()

                # Get download link from youtube video's url and init capture
                best_url = streamlink.streams(self.url)['best'].url
                self.container = av.open(
                    best_url, format='segment', timeout=(60.0, 30.0),
                    options={
                        'http_persistent' : '0',
                        'http_multiple' : '1',
                    })
                stream = self.container.streams.get(video=0)[0]
                
                # Extract keyframes only
                stream.codec_context.skip_frame = 'NONKEY'
                # Enable multiframe multithreading
                stream.thread_type = 'AUTO'

                return frame_generator(self.container, stream)

            except KeyboardInterrupt:
                raise
            
            except:
                msg = 'Could not access stream. Retrying in 60 seconds'
                self.logger.log(msg, 'ERR')
                time.sleep(60)


    def addframe(self, frame):
        """Add frame to frames dictionary

        Filename: channelname_frame_timestamp.png

        Args:
            frame (ndarray): Image as numpy array
        """
        timestamp = '{:%Y%m%d%H%M%S%f}'.format(datetime.now())
        filename = f'{self.name}_frame_{timestamp}.png'
        image = Image.fromarray(frame)

        self.frames[filename] = image


    def saveframes(self):
        """Save frames to a path defined during object initialization

        Path: output_dir/frames
        """
        self.logger.log(f'Saving {len(self.frames)} frames', 'INFO')

        with DelayedKeyboardInterrupt():
            for filename, image in self.frames.items():
                image.save(self.paths['frames'] / filename)

            self.frames = {}


    def getpaths(self, output_dir):
        """Define filepaths for storing frames and logs

        Create filepaths if they do not exist.

        Args:
            output_dir (str): Output directory

        Returns:
            dict: Dictionary of pathlib.Path objects
        """
        paths = {
            'output' : Path(output_dir),
            'frames' : Path(output_dir) / 'frames',
            'logs' : Path(output_dir) / 'logs',
        }
        for _, path in paths.items():
            path.mkdir(exist_ok=True, parents=True)

        return paths


    def getlogger(self):
        """Initialize logging object

        Path: output_dir/logs
        Filename: channelname_stream_capture_log_timestamp.txt
        """
        name = '{}_stream_capture_log_{:%Y%m%d%H%M}.txt'.format(self.name, datetime.now())
        filepath = self.paths['logs'] / name
        logger = Logging(filepath)

        logger.log('Logger Initialized', 'INFO')

        return logger


    def close(self):
        """Store data and release resources on exit

        Always executed when process exits.
        """
        self.saveframes()

        self.logger.log('Closing Stream Capture', 'INFO')
        self.logger.close()