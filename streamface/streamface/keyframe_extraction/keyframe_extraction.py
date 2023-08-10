from ..utils import detect_emptyness, detect_blur, template_matching


class KeyframeExtraction(object):
    """Extract high quality and distinct frames"""

    def __init__(self, empty_threshold, blur_threshold, similarity_threshold):
        self.empty_threshold = empty_threshold
        self.blur_threshold = blur_threshold
        self.similarity_threshold = similarity_threshold
        self.last_frame = None

    def getkeyframe(self, frame):

        empty = detect_emptyness(frame)
        if empty > self.empty_threshold:
            return None

        blur = detect_blur(frame, (1280, 720), 11, 0.15)
        if blur < self.blur_threshold:
            return None

        if self.last_frame is not None:
            sim = template_matching(frame, self.last_frame)
            if sim > self.similarity_threshold:
                return None

        self.last_frame = frame

        return frame
