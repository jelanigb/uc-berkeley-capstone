"""
Harvester v5 — with poll_label tracking, quota counting, retries disabled, and synthetic
data check.

Architecture:
  Phase 1: Scan channels for NEW videos → insert tracking + "upload" snapshot
  Phase 2: Query tracking table for videos due for "24h" poll → snapshot
  Phase 3: Query tracking table for videos due for "7d" poll → snapshot

Features:
  - poll_label tracking (upload, 24h, 7d) via videos_to_track table
  - Load jobs for tracking inserts (no streaming buffer, safe to UPDATE)
  - upload_poll_done only set TRUE after snapshot write succeeds
  - YouTube quota exceeded detection with 429 response
  - Per-run quota usage logging (calls + estimated units)
  - Retries disabled (num_retries=0) to avoid burning quota
  - Consecutive error bail-out after 20 failures
  - Category name mapping + ISO 8601 duration parsing
  - Full description storage (no truncation)
  - Subscriber count at poll time
  - Synthetic media label (status.containsSyntheticMedia)

Runs every 3 hours on Cloud Run. Default batch size: 500 channels.
Test mode: append ?limit=N to process fewer channels.
"""

import io
import json
import os
import re
import cv2
import numpy as np
import requests
from datetime import datetime
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.cloud import bigquery
from google.cloud.bigquery import LoadJobConfig, SourceFormat
import functions_framework

# --- Configuration ---
YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY")
PROJECT_ID = os.environ.get("PROJECT_ID")
DATASET_ID = "capstone_youtube"

# How many channels to process per run
DEFAULT_BATCH_SIZE = 500

# Bail out after this many consecutive YouTube errors
MAX_CONSECUTIVE_ERRORS = 20

# Poll windows — how old a video should be for each label
POLL_WINDOWS = {
    "upload": (0, 6),       # 0-6 hours old → first snapshot
    "24h":   (20, 30),      # 20-30 hours old → day-1 snapshot
    "7d":    (156, 180),    # 6.5-7.5 days old → day-7 snapshot
}

# YouTube category ID → name mapping
CATEGORY_MAP = {
    "1": "Film & Animation", "2": "Autos & Vehicles", "10": "Music",
    "15": "Pets & Animals", "17": "Sports", "19": "Travel & Events",
    "20": "Gaming", "22": "People & Blogs", "23": "Comedy",
    "24": "Entertainment", "25": "News & Politics", "26": "Howto & Style",
    "27": "Education", "28": "Science & Technology", "29": "Nonprofits & Activism",
}

# Quota cost per YouTube API method
QUOTA_COSTS = {
    "search.list": 100,
    "playlistItems.list": 1,
    "channels.list": 1,
    "videos.list": 1,
    "unknown": 1,
}

# --- Per-run quota tracking ---
api_call_counts = {}


# --- Quota Error Handling ---

class YouTubeQuotaExceeded(Exception):
    """Raised when YouTube API daily quota is exhausted."""
    pass


def check_quota_error(error):
    """Inspects a Google API HttpError and raises if it's a quota issue."""
    if isinstance(error, HttpError) and error.resp.status == 403:
        error_details = error.content.decode("utf-8", errors="replace")
        if "quotaExceeded" in error_details:
            raise YouTubeQuotaExceeded(
                f"YouTube API daily quota exceeded. "
                f"Details: {error_details[:500]}"
            )


def safe_youtube_call(api_request):
    """
    Executes a YouTube API request with:
      - Quota detection (raises YouTubeQuotaExceeded)
      - Call counting for per-run quota logging
      - Retries disabled (num_retries=0) to avoid wasting quota
    """
    uri = getattr(api_request, "uri", "unknown")
    if "playlistItems" in uri:
        method_name = "playlistItems.list"
    elif "channels" in uri:
        method_name = "channels.list"
    elif "/videos" in uri:
        method_name = "videos.list"
    elif "search" in uri:
        method_name = "search.list"
    else:
        method_name = "unknown"

    api_call_counts[method_name] = api_call_counts.get(method_name, 0) + 1

    try:
        return api_request.execute(num_retries=0)
    except HttpError as e:
        check_quota_error(e)
        raise


def log_quota_usage():
    """Logs the quota usage for this run and returns total estimated units."""
    quota_breakdown = {
        method: count * QUOTA_COSTS.get(method, 1)
        for method, count in api_call_counts.items()
    }
    total_quota = sum(quota_breakdown.values())
    total_calls = sum(api_call_counts.values())

    print(f"📊 API calls this run: {api_call_counts} ({total_calls} total)")
    print(f"📊 Estimated quota: {quota_breakdown} = {total_quota} units")

    api_call_counts.clear()
    return total_quota


# --- Helpers ---

def parse_duration(duration_str):
    """Convert ISO 8601 duration (PT15M33S) to seconds."""
    match = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", duration_str)
    if not match:
        return 0
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    seconds = int(match.group(3) or 0)
    return hours * 3600 + minutes * 60 + seconds


def analyze_thumbnail(url):
    """Downloads thumbnail and extracts CV features."""
    try:
        resp = requests.get(url, timeout=5)
        image_bytes = np.asarray(bytearray(resp.content), dtype="uint8")
        img = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        if img is None:
            return None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)

        (B, G, R) = cv2.split(img.astype("float"))
        rg = np.absolute(R - G)
        yb = np.absolute(0.5 * (R + G) - B)
        colorfulness = (
            np.sqrt(np.std(rg) ** 2 + np.std(yb) ** 2)
            + 0.3 * np.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2)
        )

        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        return {
            "brightness": float(brightness),
            "colorfulness": float(colorfulness),
            "face_count": int(len(faces)),
        }
    except Exception:
        return None


def get_channel_stats(youtube, channel_ids):
    """Fetch current subscriber counts for a batch of channel IDs."""
    if not channel_ids:
        return {}
    result = {}
    for i in range(0, len(channel_ids), 50):
        batch = channel_ids[i:i + 50]
        res = safe_youtube_call(
            youtube.channels().list(
                part="statistics", id=",".join(batch)
            )
        )
        for ch in res.get("items", []):
            result[ch["id"]] = int(ch["statistics"].get("subscriberCount", 0))
    return result


def insert_tracking_rows(bq_client, tracking_rows):
    """
    Insert rows into videos_to_track using a load job instead of streaming.
    Load jobs write directly to managed storage (no streaming buffer),
    so rows are immediately available for UPDATE.
    """
    tracking_table = f"{PROJECT_ID}.{DATASET_ID}.videos_to_track"
    load_config = LoadJobConfig(
        source_format=SourceFormat.NEWLINE_DELIMITED_JSON,
        write_disposition="WRITE_APPEND",
    )
    ndjson = "\n".join(json.dumps(r) for r in tracking_rows)
    load_job = bq_client.load_table_from_file(
        io.BytesIO(ndjson.encode("utf-8")),
        tracking_table,
        job_config=load_config,
    )
    load_job.result()  # Wait for completion
    print(f"  Loaded {len(tracking_rows)} rows into videos_to_track via load job")


def build_snapshot_row(video, channel_info, poll_label, now, sub_count=0):
    """Build a single snapshot row from YouTube API video response."""
    snippet = video["snippet"]
    stats = video.get("statistics", {})
    content = video.get("contentDetails", {})
    status = video.get("status", {})

    published_at = snippet.get("publishedAt", "")
    pub_dt = datetime.strptime(published_at, "%Y-%m-%dT%H:%M:%SZ")
    hours_since = (now - pub_dt).total_seconds() / 3600

    thumb_url = snippet["thumbnails"].get(
        "high", snippet["thumbnails"]["default"]
    )["url"]
    cv_features = analyze_thumbnail(thumb_url)

    duration_str = content.get("duration", "PT0S")
    duration_seconds = parse_duration(duration_str)
    category_id = snippet.get("categoryId", "")

    return {
        "video_id": video["id"],
        "poll_timestamp": now.isoformat(),
        "poll_label": poll_label,
        "channel_id": channel_info["channel_id"],
        "channel_handle": channel_info.get("channel_handle", ""),
        "vertical": channel_info.get("vertical", ""),
        "tier": channel_info.get("tier", ""),
        "title": snippet["title"],
        "description": snippet.get("description", ""),
        "tags": snippet.get("tags", []),  # Raw list for ARRAY<STRING> column
        "duration_seconds": duration_seconds,
        "category_id": category_id,
        "category_name": CATEGORY_MAP.get(category_id, "Unknown"),
        "published_at": published_at,
        "hours_since_publish": round(hours_since, 2),
        "view_count": int(stats.get("viewCount", 0)),
        "like_count": int(stats.get("likeCount", 0)),
        "comment_count": int(stats.get("commentCount", 0)),
        "subscriber_count": sub_count,
        "face_count": cv_features["face_count"] if cv_features else 0,
        "brightness": cv_features["brightness"] if cv_features else 0.0,
        "colorfulness": cv_features["colorfulness"] if cv_features else 0.0,
        "contains_synthetic_media": bool(status.get("containsSyntheticMedia")),
    }


def poll_videos(youtube, bq_client, video_rows, poll_label, now):
    """
    Shared logic for 24h and 7d follow-up polls.
    Takes query result rows, fetches fresh stats, returns snapshot rows.
    """
    if not video_rows:
        return []

    vid_ids = [r.video_id for r in video_rows]
    ch_map = {
        r.video_id: {
            "channel_id": r.channel_id,
            "channel_handle": r.channel_handle,
            "vertical": r.vertical,
            "tier": r.tier,
        }
        for r in video_rows
    }

    unique_channels = list({r.channel_id for r in video_rows})
    sub_counts = get_channel_stats(youtube, unique_channels)

    snapshots = []
    for i in range(0, len(vid_ids), 50):
        batch = vid_ids[i:i + 50]
        v_res = safe_youtube_call(
            youtube.videos().list(
                id=",".join(batch),
                part="snippet,statistics,contentDetails,status"
            )
        )
        for v in v_res.get("items", []):
            ch_info = ch_map[v["id"]]
            sub_count = sub_counts.get(ch_info["channel_id"], 0)
            snapshots.append(
                build_snapshot_row(v, ch_info, poll_label, now, sub_count)
            )

    # Mark polls as done
    flag_col = "day1_poll_done" if poll_label == "24h" else "day7_poll_done"
    vid_list = ", ".join(f"'{v}'" for v in vid_ids)
    bq_client.query(f"""
        UPDATE `{PROJECT_ID}.{DATASET_ID}.videos_to_track`
        SET {flag_col} = TRUE
        WHERE video_id IN ({vid_list})
    """).result()

    return snapshots


# --- Main Entry Point ---

@functions_framework.http
def run_harvester(request):
    try:
        # Override batch size for testing
        test_limit = int(request.args.get("limit", 0))
        BATCH_SIZE = test_limit if test_limit > 0 else DEFAULT_BATCH_SIZE

        if test_limit > 0:
            print(f"⚙️  TEST MODE: limit {BATCH_SIZE} channels")

        youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
        bq_client = bigquery.Client(project=PROJECT_ID)
        now = datetime.utcnow()
        all_snapshots = []

        # =============================================
        # PHASE 1: Detect new videos → "upload" snapshot
        # =============================================
        channel_query = f"""
            SELECT channel_id, channel_handle, vertical, tier
            FROM `{PROJECT_ID}.{DATASET_ID}.channels_to_track`
            ORDER BY RAND()
            LIMIT {BATCH_SIZE}
        """
        try:
            channels = list(bq_client.query(channel_query).result())
        except Exception as e:
            print(f"❌ BigQuery channel query failed: {e}")
            return f"BigQuery Query Error: {e}", 500

        if not channels:
            return "No channels returned from BigQuery", 200

        print(f"Phase 1: Scanning {len(channels)} channels for new videos...")

        new_video_candidates = []
        consecutive_errors = 0

        for ch in channels:
            try:
                playlist_id = "UU" + ch.channel_id[2:]
                res = safe_youtube_call(
                    youtube.playlistItems().list(
                        playlistId=playlist_id, part="snippet", maxResults=5
                    )
                )

                for item in res.get("items", []):
                    vid_id = item["snippet"]["resourceId"]["videoId"]
                    pub = item["snippet"].get("publishedAt", "")
                    if not pub:
                        continue

                    pub_dt = datetime.strptime(pub, "%Y-%m-%dT%H:%M:%SZ")
                    hours_old = (now - pub_dt).total_seconds() / 3600

                    lo, hi = POLL_WINDOWS["upload"]
                    if lo <= hours_old <= hi:
                        new_video_candidates.append((vid_id, pub, {
                            "channel_id": ch.channel_id,
                            "channel_handle": ch.channel_handle,
                            "vertical": ch.vertical,
                            "tier": ch.tier,
                        }))

                consecutive_errors = 0  # Reset on success

            except YouTubeQuotaExceeded:
                raise  # Always bubble up
            except Exception as e:
                consecutive_errors += 1
                print(f"Error on {ch.channel_handle}: {e}")
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    print(f"🛑 {MAX_CONSECUTIVE_ERRORS} consecutive failures — aborting Phase 1")
                    break
                continue

        # Check which are truly new
        upload_vid_ids = []  # Track which videos need upload_poll_done marked

        if new_video_candidates:
            candidate_ids = [c[0] for c in new_video_candidates]
            id_list = ", ".join(f"'{vid}'" for vid in candidate_ids)
            existing_query = f"""
                SELECT video_id FROM `{PROJECT_ID}.{DATASET_ID}.videos_to_track`
                WHERE video_id IN ({id_list})
            """
            existing = {
                r.video_id for r in bq_client.query(existing_query).result()
            }

            truly_new = [c for c in new_video_candidates if c[0] not in existing]

            if truly_new:
                # Insert into tracking table via load job (no streaming buffer)
                tracking_rows = [{
                    "video_id": vid_id,
                    "channel_id": ch_info["channel_id"],
                    "published_at": pub_at,
                    "first_seen_at": now.isoformat(),
                    "upload_poll_done": False,
                    "day1_poll_done": False,
                    "day7_poll_done": False,
                } for vid_id, pub_at, ch_info in truly_new]

                insert_tracking_rows(bq_client, tracking_rows)

                # Fetch full video details + create upload snapshots
                upload_vid_ids = [c[0] for c in truly_new]
                ch_map = {c[0]: c[2] for c in truly_new}

                unique_channels = list(
                    {info["channel_id"] for info in ch_map.values()}
                )
                sub_counts = get_channel_stats(youtube, unique_channels)

                for i in range(0, len(upload_vid_ids), 50):
                    batch = upload_vid_ids[i:i + 50]
                    v_res = safe_youtube_call(
                        youtube.videos().list(
                            id=",".join(batch),
                            part="snippet,statistics,contentDetails,status"
                        )
                    )
                    for v in v_res.get("items", []):
                        ch_info = ch_map[v["id"]]
                        sub_count = sub_counts.get(ch_info["channel_id"], 0)
                        row = build_snapshot_row(
                            v, ch_info, "upload", now, sub_count
                        )
                        all_snapshots.append(row)

        print(f"Phase 1 complete: {len(new_video_candidates)} candidates, "
              f"{sum(1 for s in all_snapshots if s['poll_label'] == 'upload')} upload snapshots")

        # =============================================
        # PHASE 2: Follow-up "24h" polls
        # =============================================
        lo24, hi24 = POLL_WINDOWS["24h"]
        day1_query = f"""
            SELECT vt.video_id, vt.channel_id, vt.published_at,
                   ct.channel_handle, ct.vertical, ct.tier
            FROM `{PROJECT_ID}.{DATASET_ID}.videos_to_track` vt
            JOIN `{PROJECT_ID}.{DATASET_ID}.channels_to_track` ct
              ON vt.channel_id = ct.channel_id
            WHERE vt.upload_poll_done = TRUE
              AND vt.day1_poll_done = FALSE
              AND TIMESTAMP_DIFF(CURRENT_TIMESTAMP(), vt.published_at, HOUR)
                  BETWEEN {lo24} AND {hi24}
            LIMIT 100
        """
        try:
            day1_videos = list(bq_client.query(day1_query).result())
        except Exception as e:
            print(f"Day1 query error: {e}")
            day1_videos = []

        all_snapshots.extend(
            poll_videos(youtube, bq_client, day1_videos, "24h", now)
        )
        print(f"Phase 2 complete: {len(day1_videos)} videos polled at 24h")

        # =============================================
        # PHASE 3: Follow-up "7d" polls
        # =============================================
        lo7, hi7 = POLL_WINDOWS["7d"]
        day7_query = f"""
            SELECT vt.video_id, vt.channel_id, vt.published_at,
                   ct.channel_handle, ct.vertical, ct.tier
            FROM `{PROJECT_ID}.{DATASET_ID}.videos_to_track` vt
            JOIN `{PROJECT_ID}.{DATASET_ID}.channels_to_track` ct
              ON vt.channel_id = ct.channel_id
            WHERE vt.day1_poll_done = TRUE
              AND vt.day7_poll_done = FALSE
              AND TIMESTAMP_DIFF(CURRENT_TIMESTAMP(), vt.published_at, HOUR)
                  BETWEEN {lo7} AND {hi7}
            LIMIT 100
        """
        try:
            day7_videos = list(bq_client.query(day7_query).result())
        except Exception as e:
            print(f"Day7 query error: {e}")
            day7_videos = []

        all_snapshots.extend(
            poll_videos(youtube, bq_client, day7_videos, "7d", now)
        )
        print(f"Phase 3 complete: {len(day7_videos)} videos polled at 7d")

        # =============================================
        # WRITE ALL SNAPSHOTS
        # =============================================
        snapshot_write_ok = False

        if all_snapshots:
            print(f"Writing {len(all_snapshots)} snapshots to BigQuery...")
            print(f"Sample row: {all_snapshots[0]}")

            table_ref = f"{PROJECT_ID}.{DATASET_ID}.video_snapshots"
            errors = bq_client.insert_rows_json(table_ref, all_snapshots)
            if errors:
                print(f"❌ BQ insert errors: {errors[:3]}")
                # Log quota even on failure
                log_quota_usage()
                return f"BQ Errors: {errors}", 500
            else:
                snapshot_write_ok = True

        # Only mark upload_poll_done after snapshots successfully written
        if snapshot_write_ok and upload_vid_ids:
            vid_list = ", ".join(f"'{v}'" for v in upload_vid_ids)
            bq_client.query(f"""
                UPDATE `{PROJECT_ID}.{DATASET_ID}.videos_to_track`
                SET upload_poll_done = TRUE
                WHERE video_id IN ({vid_list})
            """).result()
            print(f"  Marked {len(upload_vid_ids)} videos as upload_poll_done")

        upload_ct = sum(1 for s in all_snapshots if s["poll_label"] == "upload")
        day1_ct = sum(1 for s in all_snapshots if s["poll_label"] == "24h")
        day7_ct = sum(1 for s in all_snapshots if s["poll_label"] == "7d")

        # Log quota usage
        quota_used = log_quota_usage()

        msg = (
            f"Success: {len(all_snapshots)} snapshots "
            f"(upload={upload_ct}, 24h={day1_ct}, 7d={day7_ct}) | "
            f"Quota: ~{quota_used} units"
        )
        print(msg)
        return msg, 200

    except YouTubeQuotaExceeded as e:
        log_quota_usage()
        print(f"🚨 QUOTA EXCEEDED: {e}")
        return f"Quota exceeded: {e}", 429

    except Exception as e:
        log_quota_usage()
        print(f"❌ Unexpected error: {e}")
        return f"Server error: {e}", 500