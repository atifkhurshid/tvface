from streamface.stream_capture import StreamCapture

"""
NOTE: One Process can only capture stream from one url.
      To capture multiple streams, start separate processes
      in Windows Terminal tabs for each config
"""

capture = StreamCapture(
    name='skynews',
    url='https://www.youtube.com/watch?v=9Auq9mYxFEE',
    output_dir='./data/skynews',
    batch_size=50,
    empty_threshold=0.95,
    blur_threshold=50,
    similarity_threshold=0.9,
    reconnect_interval=500,
    log_interval=100
)

capture.call()
