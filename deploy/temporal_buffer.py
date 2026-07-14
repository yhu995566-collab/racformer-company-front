"""Eight-frame buffering independent of ROS and dataset implementations."""

from collections import deque

from .input_schema import FrameInput


class TemporalFrameBuffer:
    """Store newest observations and expose current-to-oldest model order."""

    def __init__(self, num_frames=8):
        if num_frames <= 0:
            raise ValueError('num_frames must be positive')
        self.num_frames = int(num_frames)
        self._frames = deque(maxlen=self.num_frames)

    def __len__(self):
        return len(self._frames)

    @property
    def ready(self):
        return len(self._frames) == self.num_frames

    def clear(self):
        self._frames.clear()

    def append(self, frame):
        if not isinstance(frame, FrameInput):
            raise TypeError('frame must be a FrameInput')
        if self._frames:
            newest = self._frames[-1]
            if frame.image_timestamp < newest.image_timestamp:
                raise ValueError('image timestamps must be non-decreasing')
            if frame.radar_timestamp < newest.radar_timestamp:
                raise ValueError('radar timestamps must be non-decreasing')
        self._frames.append(frame)

    def snapshot(self, pad=True):
        """Return newest first, matching current + historical sweep order."""
        if not self._frames:
            raise RuntimeError('temporal buffer is empty')
        frames = list(reversed(self._frames))
        if pad and len(frames) < self.num_frames:
            frames.extend([frames[-1]] * (self.num_frames - len(frames)))
        if len(frames) != self.num_frames:
            raise RuntimeError(
                'buffer has {} frames, expected {}'.format(
                    len(frames), self.num_frames))
        return frames
