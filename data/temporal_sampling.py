import os
import random
from collections import OrderedDict


def validate_temporal_frame_step_random_max(frame_step, random_max):
    if random_max > 0 and random_max < frame_step:
        raise ValueError(
            "--data_temporal_frame_step_random_max must be 0 or >= "
            "--data_temporal_frame_step"
        )


def temporal_valid_start_count(num_paths, num_frames, frame_step):
    return num_paths - (num_frames - 1) * frame_step


def build_temporal_series_index(paths, num_frames, frame_step):
    vid_series_paths = list(
        OrderedDict.fromkeys(os.path.dirname(path) for path in paths)
    )
    raw_counts = {vid_serie: 0 for vid_serie in vid_series_paths}
    for path in paths:
        raw_counts[os.path.dirname(path)] += 1

    frames_counts = {
        vid_serie: temporal_valid_start_count(
            raw_counts[vid_serie], num_frames, frame_step
        )
        for vid_serie in vid_series_paths
    }

    cumulative_sums = []
    cumulative_sum = 0
    for vid_serie in vid_series_paths:
        count = frames_counts[vid_serie]
        if count > 0:
            cumulative_sum += count
        cumulative_sums.append(cumulative_sum)

    available_frame_pool = []
    start_count = 0
    for i in range(len(cumulative_sums)):
        num_frames_available = (
            cumulative_sums[i] - cumulative_sums[i - 1]
            if i != 0
            else cumulative_sums[i]
        )
        end_count = start_count + num_frames_available
        available_frame_pool.append(list(range(start_count, end_count)))
        start_count = end_count

    return vid_series_paths, frames_counts, cumulative_sums, available_frame_pool


def select_temporal_start_from_series(paths, series_index):
    vid_series_paths, _frames_counts, cumulative_sums, available_frame_pool = (
        series_index
    )
    if not cumulative_sums or cumulative_sums[-1] <= 0:
        return None

    random_start = random.randint(0, cumulative_sums[-1] - 1)
    selected_index = [
        i
        for i, frame_pool in enumerate(available_frame_pool)
        if random_start in frame_pool
    ]
    if len(selected_index) != 1:
        raise ValueError("random temporal start not found in any video series")
    selected_index = selected_index[0]
    selected_vid = vid_series_paths[selected_index]
    if selected_index > 0:
        frame_num = random_start - cumulative_sums[selected_index - 1]
    else:
        frame_num = random_start

    filtered_paths = [path for path in paths if os.path.dirname(path) == selected_vid]
    selected_path = filtered_paths[frame_num]
    return paths.index(selected_path)


class TemporalFrameStepMixin:
    def _init_temporal_frame_step_sampling(self, opt):
        self.num_frames = opt.data_temporal_number_frames
        self.frame_step = opt.data_temporal_frame_step
        self.frame_step_random_max = int(
            getattr(opt, "data_temporal_frame_step_random_max", 0) or 0
        )
        validate_temporal_frame_step_random_max(
            self.frame_step, self.frame_step_random_max
        )

    def _random_temporal_frame_step_enabled(self):
        return int(getattr(self, "frame_step_random_max", 0) or 0) > 0

    def _sample_temporal_frame_step(self):
        if self._random_temporal_frame_step_enabled():
            return random.randint(self.frame_step, self.frame_step_random_max)
        return self.frame_step

    def _select_single_temporal_start(self, num_paths, frame_step):
        start_count = temporal_valid_start_count(num_paths, self.num_frames, frame_step)
        if start_count <= 0:
            return None
        return random.randint(0, start_count - 1)
