import pytest

from data.self_supervised_vid_labeled_mask_cls_online_dataset import (
    SelfSupervisedVidLabeledMaskClsOnlineDataset,
)
from data.self_supervised_vid_mask_online_dataset import (
    SelfSupervisedVidMaskOnlineDataset,
)
from data.temporal_sampling import build_temporal_series_index


@pytest.mark.parametrize(
    "dataset_class",
    [
        SelfSupervisedVidMaskOnlineDataset,
        SelfSupervisedVidLabeledMaskClsOnlineDataset,
    ],
)
def test_temporal_test_sampling_visits_every_valid_window_once(dataset_class):
    paths = [
        "video_a/frame_000.jpg",
        "video_a/frame_001.jpg",
        "video_a/frame_002.jpg",
        "video_b/frame_000.jpg",
        "video_b/frame_001.jpg",
        "video_b/frame_002.jpg",
        "video_b/frame_003.jpg",
    ]
    series_index = build_temporal_series_index(paths, num_frames=2, frame_step=1)

    dataset = object.__new__(dataset_class)
    dataset.phase = "test"
    dataset.frame_step_random_max = 0
    dataset.A_img_paths = paths
    dataset.A_size = len(paths)
    (
        dataset.vid_series_paths,
        dataset.frames_counts,
        dataset.cumulative_sums,
        dataset.available_frame_pool,
    ) = series_index

    assert len(dataset) == 5
    assert [
        dataset._select_temporal_index_A(frame_step=1, dataset_index=index)
        for index in range(len(dataset))
    ] == [0, 1, 3, 4, 5]


def test_temporal_test_sampling_rejects_index_past_valid_windows():
    paths = ["video/frame_000.jpg", "video/frame_001.jpg"]
    series_index = build_temporal_series_index(paths, num_frames=2, frame_step=1)

    dataset = object.__new__(SelfSupervisedVidMaskOnlineDataset)
    dataset.phase = "test"
    dataset.frame_step_random_max = 0
    dataset.A_img_paths = paths
    dataset.A_size = len(paths)
    (
        dataset.vid_series_paths,
        dataset.frames_counts,
        dataset.cumulative_sums,
        dataset.available_frame_pool,
    ) = series_index

    with pytest.raises(IndexError, match="outside"):
        dataset._select_temporal_index_A(frame_step=1, dataset_index=1)
