import json
import sys

import cv2
import numpy as np
import pytest
import torch

from util import evaluate_fvid as fvd


@pytest.fixture(autouse=True)
def limit_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def write_video(path, count, color):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (16, 16))
    assert writer.isOpened()
    for _ in range(count):
        writer.write(np.full((16, 16, 3), color, dtype=np.uint8))
    writer.release()


def test_rgb_layout_and_tail(tmp_path):
    path = tmp_path / "red.mp4"
    write_video(path, 7, (0, 0, 255))
    info = {}
    batches = list(fvd.video_batches(path, clip_length=3, batch_size=4, info=info))
    assert batches[0].shape == (2, 3, 3, 224, 224)
    assert batches[0][:, 0].mean() > 0.9  # Red, not blue.
    assert batches[0][:, 2].mean() < -0.9
    assert info["clips"] == 2
    assert info["discarded_frames"] == 1


class FakeBackbone:
    def __init__(self, name):
        self.name = name

    def extract_features(self, x):
        dims = 1024 if self.name == "i3d" else 2048
        vector = x.mean((1, 2, 3, 4))[:, None].expand(-1, dims)
        return vector[:, :, None, None, None] if self.name == "i3d" else vector


@pytest.mark.parametrize("name", ["i3d", "resnext"])
def test_batch_consistency_and_video_boundaries(tmp_path, monkeypatch, name):
    videos = tmp_path / "videos"
    videos.mkdir()
    write_video(videos / "red.mp4", 7, (0, 0, 255))
    write_video(videos / "white.mp4", 7, (255, 255, 255))
    monkeypatch.setattr(fvd, "build_model", lambda name, *_: FakeBackbone(name))

    def collect(batch_size):
        return fvd.collect_features(
            videos,
            [name],
            {name: tmp_path / "weights"},
            3,
            batch_size,
            "cpu",
            {},
        )

    first, info = collect(1)
    second, _ = collect(4)
    np.testing.assert_allclose(first[name], second[name], atol=1e-6)
    assert info["clip_count"] == 4  # Each video's remaining frame is discarded.
    assert info["discarded_frames"] == 2


def test_frechet_identical_and_shifted_features():
    features = np.random.default_rng(42).normal(size=(20, 8))
    assert fvd.calculate_fid(features, features.copy()) == pytest.approx(0, abs=1e-10)
    assert fvd.calculate_fid(features, features + 1) == pytest.approx(8, abs=1e-10)


def test_incomplete_decode_is_error(tmp_path, monkeypatch):
    class BrokenCapture:
        def isOpened(self):
            return True

        def get(self, prop):
            return 10 if prop == cv2.CAP_PROP_FRAME_COUNT else 25

        def read(self):
            return False, None

        def release(self):
            pass

    monkeypatch.setattr(cv2, "VideoCapture", lambda _: BrokenCapture())
    with pytest.raises(ValueError, match="Incomplete decode"):
        list(fvd.video_batches(tmp_path / "broken.mp4", 3, 2, {}))


def test_too_few_clips(tmp_path):
    write_video(tmp_path / "short.mp4", 1, (0, 0, 255))
    with pytest.raises(ValueError, match="at least two complete clips"):
        fvd.collect_features(tmp_path, ["i3d"], {}, 10, 4, "cpu", {})


@pytest.mark.parametrize("automatic_weights", [False, True])
@pytest.mark.parametrize("file_inputs", [False, True])
def test_cli_result(tmp_path, monkeypatch, automatic_weights, file_inputs):
    monkeypatch.chdir(tmp_path)
    real, generated, weights = [
        tmp_path / name for name in ("real", "generated", "weights")
    ]
    for directory in (real, generated, weights):
        directory.mkdir()
    for directory in (real, generated):
        write_video(directory / "video.mp4", 7, (0, 0, 255))
    for filename in fvd.WEIGHTS.values():
        (weights / filename).write_bytes(b"test weights; loader is replaced")
    monkeypatch.setattr(fvd.torch.hub, "get_dir", lambda: str(tmp_path / "hub"))
    downloaded = []
    if automatic_weights:

        def download(name, path):
            downloaded.append(name)
            assert path == tmp_path / "hub" / "checkpoints" / "fvd" / fvd.WEIGHTS[name]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"downloaded test weights; loader is replaced")

        monkeypatch.setattr(fvd, "ensure_weights", download)
    monkeypatch.setattr(fvd, "build_model", lambda name, *_: FakeBackbone(name))
    original_distance = fvd.calculate_fid
    monkeypatch.setattr(
        fvd, "calculate_fid", lambda a, b: original_distance(a[:, :4], b[:, :4])
    )
    output = (
        tmp_path / "tmp" / "fvd_results.json"
        if automatic_weights
        else tmp_path / "result.json"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_fvid.py",
            "--real-dir",
            str(real / "video.mp4" if file_inputs else real),
            "--generated-dir",
            str(generated / "video.mp4" if file_inputs else generated),
            *([] if automatic_weights else ["--weights-dir", str(weights)]),
            "--clip-length",
            "3",
            "--device",
            "cpu",
            *(
                []
                if automatic_weights
                else [
                    "--output",
                    str(output),
                ]
            ),
        ],
    )
    fvd.main()
    result = json.loads(output.read_text())
    assert result["scores"] == pytest.approx(
        {"fvd_i3d": 0, "fvd_resnext": 0}, abs=1e-10
    )
    assert result["real"]["clip_count"] == result["generated"]["clip_count"] == 2
    assert result["settings"]["clip_length"] == 3
    assert len(result["settings"]["weight_sha256"]["i3d"]) == 64
    assert downloaded == (list(fvd.WEIGHTS) if automatic_weights else [])


@pytest.mark.parametrize("failure", [None, "html", "truncated", "network"])
def test_weight_download_and_reuse(tmp_path, monkeypatch, failure):
    payload = b"checkpoint bytes"

    class Response:
        headers = {
            "Content-Type": (
                "text/html" if failure == "html" else "application/octet-stream"
            ),
            "Content-Length": str(len(payload) + (1 if failure == "truncated" else 0)),
        }

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            yield payload
            if failure == "network":
                raise fvd.requests.ConnectionError("interrupted")

    calls = []

    def get(url, **kwargs):
        calls.append(url)
        assert kwargs["stream"] is True
        return Response()

    monkeypatch.setattr(fvd.requests, "get", get)
    path = tmp_path / "weights" / "i3d.pt"
    if failure:
        with pytest.raises(RuntimeError, match="Cannot download i3d weights"):
            fvd.ensure_weights("i3d", path)
        assert not path.exists()
    else:
        fvd.ensure_weights("i3d", path)
        assert path.read_bytes() == payload
        fvd.ensure_weights("i3d", path)
    assert calls == [fvd.WEIGHT_URLS["i3d"]]
    assert list(path.parent.glob("*.part")) == []
