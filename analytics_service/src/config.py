# Configuration settings
import os

import torch


def _select_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
_ANALYTICS_SERVICE_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))

LOCAL_VIDEO_DIR = os.path.join(_REPO_ROOT, "data", "raw")  # Local footage the job API may read
VIDEO_SOURCE = os.path.join(LOCAL_VIDEO_DIR, "samplevideo.mp4")  # Path to CCTV footage
OUTPUT_CSV = os.path.join(_ANALYTICS_SERVICE_DIR, "results", "analytics_table.csv")  # Output file path
FRAME_SKIP = 5  # Process every 5th frame for performance
MIN_CONFIDENCE = 0.5  # Minimum detection confidence
TRACKER_CONFIG = "bytetrack.yaml"  # Ultralytics tracker config (ByteTrack)

# Detector weights - override via ANALYTICS_DETECTION_WEIGHTS to A/B a
# fine-tuned checkpoint against the stock COCO weights without code changes.
# NOTE: analytics_service/finetune_mannequin_filter.py fine-tunes YOLOv8n on
# hard-negative mannequin photos to reduce false "person" hits, but the
# checkpoint produced from the initial ~45-photo Commons dataset collapsed
# detection confidence to near-zero on real people too (see the fine-tune
# plan's evaluate step) - it is NOT safe to point this at yet. Re-run
# finetune_mannequin_filter.py with a much larger/more balanced dataset
# before trying this env var against production footage.
DETECTION_WEIGHTS = os.environ.get("ANALYTICS_DETECTION_WEIGHTS", "yolov8n.pt")

# Re-identification (OSNet via torchreid)
# osnet_ain_x1_0/msmt17 (was osnet_x1_0/market1501): AIN's instance-normalized
# layers plus MSMT17's larger, more viewpoint/domain-diverse training set give
# much better cross-domain generalization than the old Market1501 checkpoint,
# which overfits hard to Market1501-like conditions and degrades sharply on
# footage that doesn't look like it (real store CCTV, for instance).
REID_MODEL_NAME = "osnet_ain_x1_0"
REID_MODEL_CHECKPOINT = "osnet_ain_x1_0_msmt17.pt"  # msmt17-trained AIN weights, auto-downloaded on first run
REID_WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")
REID_THRESHOLD = 0.94  # Cosine similarity threshold for re-identification, re-tuned for
                       # osnet_ain_x1_0/msmt17 via validate_pipeline.py on data/raw/samplevideo6.mp4
                       # (dense, frame-by-frame: same-person p5=0.900, median=0.984; different-person
                       # p95=0.638, median=0.500 - a much cleaner separation than the old
                       # osnet_x1_0/market1501 checkpoint had, 0.900 vs 0.638 here vs 0.95 vs 0.76
                       # before). Swept 0.80-0.94 against one collected detection set: false merges
                       # stay at 0 for every threshold >=0.92 and jump to 506 at 0.90, so 0.92 is the
                       # tightest safe value - it catches one more real track merge than 0.94 (231 vs
                       # 232 identities from 235 tracks) with the same zero false-merge rate.
                       # NOTE: tuned on dense, frame-by-frame footage (consecutive same-person
                       # detections a fraction of a second apart). The sparse-sampling path used
                       # for big videos (render_tracked_video.py/analytics_api.py with
                       # sample_frames > 0) sees consecutive appearances seconds apart, so real
                       # same-person similarity is naturally lower - reusing this default there
                       # fragments one person into several new identities and inflates footfall.
                       # Re-validate with `validate_pipeline.py --sample-frames --sample-window-seconds`
                       # matching your sampling and pass the result via --reid-threshold / the
                       # analysis job's reid_threshold field.

REID_THRESHOLD_SPARSE = 0.85  # Default for analytics_api.py's 3-frames/10s sampling.
                       # Re-measured for osnet_ain_x1_0/msmt17 via
                       # `validate_pipeline.py --sample-frames 3 --sample-window-seconds 10` on
                       # data/raw/samplevideo6.mp4: same-person sim p5=0.489, median=0.819;
                       # different-person p95=0.648, median=0.501. The distributions overlap badly
                       # here - notably *worse* separation than the old osnet_x1_0/market1501
                       # checkpoint had at this same sparse setting (which saw same-p5=0.61 vs
                       # diff-p95=0.75). Swept 0.60-0.90: every value below 0.85 produces double-digit
                       # to 500+ false merges (e.g. 0.80 -> 24 false merges for only 5 real merges),
                       # and 0.85/0.90 both land at 0 false merges but ALSO 0 real merges (37/37,
                       # i.e. no better than skipping re-id entirely on this clip). Kept at 0.85 as
                       # the "do no harm" choice, but this is a genuinely weaker result than dense
                       # mode and than the old checkpoint's sparse behavior - osnet_ain_x1_0/msmt17's
                       # embeddings may be more sensitive to the pose/lighting drift that accumulates
                       # over a multi-second sampling gap. Don't trust this single-video sweep as
                       # final: re-validate on more/longer real footage (more tracks and identities
                       # than this clip's 37) before relying on sparse re-id merging in production,
                       # and inspect results/reid_validation_log.csv for _FALSE_MERGE rows.

# Short-lived cross-process/cross-camera re-identification store.
CHROMADB_HOST = os.environ.get("CHROMADB_HOST", "localhost")
CHROMADB_PORT = int(os.environ.get("CHROMADB_PORT", "8000"))
CHROMADB_TENANT = os.environ.get("CHROMADB_TENANT", "default_tenant")
CHROMADB_DATABASE = os.environ.get("CHROMADB_DATABASE", "default_database")
CHROMADB_COLLECTION = os.environ.get("CHROMADB_COLLECTION", "person_reid")
CHROMADB_TTL_HOURS = int(os.environ.get("CHROMADB_TTL_HOURS", "48"))

AGE_GENDER_MODEL = "MiVOLO"  # iitolstykh/mivolo_v2 via transformers, see demographics.py
DEMOGRAPHICS_MODEL = "body"  # Options: 'face', 'body', 'hybrid' - only 'body' is implemented
# Reject weak MiVOLO reads: no new identity / no footfall entry below this.
# 0.80 balances recall vs precision on results/demographics_labels.csv (the
# labeled spot-check set): 0.97 only keeps 48% of real visitors (100%
# precision) vs 0.80's 86% recall at 83% precision. Was accidentally set to
# 0.97 in the commit that added tests/test_identity_confidence.py, which
# still asserts 0.80 - keep this in sync with that test.
MIN_DEMOGRAPHICS_CONFIDENCE = 0.80

# Static-object filtering: a resolved identity (post re-id, so ByteTrack
# fragments across sample-window gaps are already merged) that sits in
# roughly the same spot for a large chunk of the clip is almost certainly a
# mannequin or a printed/on-screen image of a person, not a live shopper -
# YOLO+ByteTrack+OSNet have no notion of "alive" and will happily detect,
# track, and re-id a completely static object. See static_objects.py.
# These are initial heuristic defaults, NOT yet validated against real
# footage - re-tune against actual store clips with known mannequins/posters
# the same way REID_THRESHOLD_SPARSE above was tuned via validate_pipeline.py,
# by inspecting the "filtering" section of the job's metrics report.
STATIC_OBJECT_MIN_SPAN_SECONDS = 45          # must persist across this much of the clip
STATIC_OBJECT_MAX_DISPLACEMENT_RATIO = 0.03  # foot-point bounding extent / frame diagonal
STATIC_OBJECT_MIN_APPEARANCES = 4            # need enough samples for a meaningful signal

# Mirror/reflection filtering: off by default, since it needs a MirrorNet
# checkpoint that isn't vendored here (see vendor/mirrornet/ and
# mirror_segmentation.py) and hasn't been validated against real store
# footage yet. A real person standing near a mirror produces two YOLO boxes
# in the same frame - the identity-merge safeguard in identity.py
# (match_identity's exclude_ids) exists specifically to stop two
# simultaneously-visible boxes from merging, which means a reflection can
# never merge back into the real person's identity and instead mints a new
# one, inflating footfall. This filters it back out post-hoc: segment each
# fixed camera's mirror region once (cached), then drop any identity that
# sits mostly inside that region AND moves in lockstep with a concurrently
# visible identity mostly outside it. See mirror_segmentation.py /
# reflection_filter.py.
MIRROR_FILTER_ENABLED = os.environ.get("ANALYTICS_MIRROR_FILTER", "0") == "1"
MIRRORNET_WEIGHTS_PATH = os.environ.get(
    "ANALYTICS_MIRRORNET_WEIGHTS", os.path.join(REID_WEIGHTS_DIR, "MirrorNet.pth")
)  # download from github.com/Mhaiyang/ICCV2019_MirrorNet - not auto-fetched
MIRRORNET_INPUT_SCALE = 384  # matches the reference repo's infer.py 'scale'
MIRROR_MASK_CACHE_DIR = os.path.join(_ANALYTICS_SERVICE_DIR, "cache", "mirror_masks")
MIRROR_CALIBRATION_FRAMES = 12    # frames sampled to build/verify a camera's mirror ROI once
MIRROR_MASK_AGG_THRESHOLD = 0.5   # fraction of calibration frames a pixel must read "mirror" in
MIRROR_MIN_APPEARANCES = 4        # same rationale as STATIC_OBJECT_MIN_APPEARANCES
MIRROR_OVERLAP_THRESHOLD = 0.6    # mean bbox-inside-mirror-mask overlap to flag a reflection candidate
MIRROR_REAL_OVERLAP_MAX = 0.2     # the real counterpart must sit mostly outside the mirror
MIRROR_APPEARANCE_SIMILARITY_THRESHOLD = 0.75  # looser than REID_THRESHOLD - mirror glass
                                                # distorts color/texture enough that a reflection's
                                                # OSNet embedding often won't clear the merge bar
MIRROR_MOTION_CORRELATION_THRESHOLD = 0.5      # foot-point displacement correlation, real vs candidate
MIRROR_MIN_SHARED_FRAMES = 3                   # frames both identities must co-appear in to compare motion
# None of the MIRROR_* thresholds above have been validated against real
# footage with a known mirror (no such clip exists in data/raw/ yet) - treat
# them as initial heuristic defaults to re-tune once one is collected, the
# same way STATIC_OBJECT_* above still needs real-footage validation.

VERTEX_PROJECT_ID = "visual-similarity-459311"
VERTEX_PROJECT_LOCATION = "us-central1"