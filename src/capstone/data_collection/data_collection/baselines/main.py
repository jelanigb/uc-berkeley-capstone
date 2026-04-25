"""
Baseline Harvester — Standalone Cloud Run function.

Gathers the last 30 videos (lifetime stats) per channel for baseline
median computation. Excludes any video already in videos_to_track to
prevent data leakage.

Deploy as a separate Cloud Run service, invoke manually or via HTTP.
  ?limit=N        Process only N channels (test mode)
  ?force=true     Re-poll channels that already have baselines

Quota cost: ~3 units per channel (1 playlistItems + 1 videos + amortized channels).
"""

import os
import re
from datetime import datetime
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.cloud import bigquery
import functions_framework

# --- Configuration ---
YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY")
PROJECT_ID = os.environ.get("PROJECT_ID")
DATASET_ID = "capstone_youtube"

DEFAULT_BATCH_SIZE = 500
MAX_CONSECUTIVE_ERRORS = 20

CATEGORY_MAP = {
    "1": "Film & Animation", "2": "Autos & Vehicles", "10": "Music",
    "15": "Pets & Animals", "17": "Sports", "19": "Travel & Events",
    "20": "Gaming", "22": "People & Blogs", "23": "Comedy",
    "24": "Entertainment", "25": "News & Politics", "26": "Howto & Style",
    "27": "Education", "28": "Science & Technology", "29": "Nonprofits & Activism",
}

QUOTA_COSTS = {
    "playlistItems.list": 1,
    "channels.list": 1,
    "videos.list": 1,
    "unknown": 1,
}

api_call_counts = {}


# --- Quota handling ---

class YouTubeQuotaExceeded(Exception):
    pass


def check_quota_error(error):
    if isinstance(error, HttpError) and error.resp.status == 403:
        error_details = error.content.decode("utf-8", errors="replace")
        if "quotaExceeded" in error_details:
            raise YouTubeQuotaExceeded(
                f"YouTube API daily quota exceeded. Details: {error_details[:500]}"
            )


def safe_youtube_call(api_request):
    uri = getattr(api_request, "uri", "unknown")
    if "playlistItems" in uri:
        method_name = "playlistItems.list"
    elif "channels" in uri:
        method_name = "channels.list"
    elif "/videos" in uri:
        method_name = "videos.list"
    else:
        method_name = "unknown"

    api_call_counts[method_name] = api_call_counts.get(method_name, 0) + 1

    try:
        return api_request.execute(num_retries=0)
    except HttpError as e:
        check_quota_error(e)
        raise


def log_quota_usage():
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
    match = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", duration_str)
    if not match:
        return 0
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    seconds = int(match.group(3) or 0)
    return hours * 3600 + minutes * 60 + seconds


def get_channel_stats(youtube, channel_ids):
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


# --- Core logic (standalone, testable) ---

def collect_channel_baselines(youtube, bq_client, channels, now,
                              project_id, dataset_id):
    """
    Gather last 30 videos (lifetime stats) per channel, excluding tracked videos.

    Args:
        youtube:    Authenticated YouTube API client
        bq_client:  BigQuery client
        channels:   List of objects with .channel_id, .channel_handle,
                    .vertical, .tier
        now:        datetime — timestamp for this poll
        project_id: GCP project ID
        dataset_id: BigQuery dataset ID

    Returns:
        dict with rows_written, channels_done, channels_skipped,
        quota_used, errors
    """
    if not channels:
        return {
            "rows_written": 0, "channels_done": 0,
            "channels_skipped": 0, "quota_used": 0, "errors": [],
        }

    # 1. Get tracked video IDs to exclude from baseline
    try:
        tracked_query = f"""
            SELECT video_id
            FROM `{project_id}.{dataset_id}.videos_to_track`
        """
        tracked_ids = {
            r.video_id for r in bq_client.query(tracked_query).result()
        }
        print(f"  Excluding {len(tracked_ids)} already-tracked video IDs")
    except Exception as e:
        print(f"  ⚠️  Could not query videos_to_track ({e}), proceeding without exclusions")
        tracked_ids = set()

    # 2. Fetch subscriber counts in bulk
    channel_ids = [ch.channel_id for ch in channels]
    sub_counts = get_channel_stats(youtube, channel_ids)

    # 3. Per channel: get last 30 non-tracked videos with full details
    all_rows = []
    consecutive_errors = 0
    channels_skipped = 0

    for ch in channels:
        try:
            playlist_id = "UU" + ch.channel_id[2:]
            pl_res = safe_youtube_call(
                youtube.playlistItems().list(
                    playlistId=playlist_id,
                    part="snippet",
                    maxResults=50,  # extra headroom for exclusions
                )
            )

            items = pl_res.get("items", [])
            if not items:
                channels_skipped += 1
                consecutive_errors = 0
                continue

            vid_ids = [
                item["snippet"]["resourceId"]["videoId"]
                for item in items
                if item["snippet"]["resourceId"]["videoId"] not in tracked_ids
            ][:30]

            if not vid_ids:
                channels_skipped += 1
                consecutive_errors = 0
                continue

            v_res = safe_youtube_call(
                youtube.videos().list(
                    id=",".join(vid_ids),
                    part="snippet,statistics,contentDetails",
                )
            )

            sub_count = sub_counts.get(ch.channel_id, 0)

            for v in v_res.get("items", []):
                snippet = v["snippet"]
                stats = v.get("statistics", {})
                content = v.get("contentDetails", {})
                duration_str = content.get("duration", "PT0S")
                category_id = snippet.get("categoryId", "")

                all_rows.append({
                    "channel_id": ch.channel_id,
                    "channel_handle": ch.channel_handle,
                    "vertical": ch.vertical,
                    "tier": ch.tier,
                    "video_id": v["id"],
                    "title": snippet.get("title", ""),
                    "published_at": snippet.get("publishedAt", ""),
                    "duration_seconds": parse_duration(duration_str),
                    "category_id": category_id,
                    "category_name": CATEGORY_MAP.get(category_id, "Unknown"),
                    "view_count": int(stats.get("viewCount", 0)),
                    "like_count": int(stats.get("likeCount", 0)),
                    "comment_count": int(stats.get("commentCount", 0)),
                    "subscriber_count": sub_count,
                    "baseline_polled_at": now.isoformat(),
                })

            consecutive_errors = 0

        except YouTubeQuotaExceeded:
            raise
        except Exception as e:
            consecutive_errors += 1
            channels_skipped += 1
            print(f"Error on {ch.channel_handle}: {e}")
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                print(f"🛑 {MAX_CONSECUTIVE_ERRORS} consecutive failures — aborting baseline")
                break
            continue

    # 4. Write to BigQuery
    result = {
        "rows_written": 0, "channels_done": 0,
        "channels_skipped": channels_skipped, "quota_used": 0, "errors": [],
    }

    if not all_rows:
        result["quota_used"] = log_quota_usage()
        return result

    print(f"Writing {len(all_rows)} baseline video rows to BigQuery...")

    table_ref = f"{project_id}.{dataset_id}.channel_baseline_videos"
    bq_errors = bq_client.insert_rows_json(table_ref, all_rows)
    result["quota_used"] = log_quota_usage()

    if bq_errors:
        print(f"❌ BQ insert errors: {bq_errors[:3]}")
        result["errors"] = [str(e) for e in bq_errors[:5]]
        return result

    result["rows_written"] = len(all_rows)
    result["channels_done"] = len(set(r["channel_id"] for r in all_rows))
    print(
        f"Baseline complete: {result['rows_written']} videos across "
        f"{result['channels_done']} channels"
    )
    return result


# --- Cloud Run entry point ---

@functions_framework.http
def run_baseline(request):
    try:
        test_limit = int(request.args.get("limit", 0))
        force = request.args.get("force", "false").lower() == "true"
        batch_size = test_limit if test_limit > 0 else DEFAULT_BATCH_SIZE

        if test_limit > 0:
            print(f"⚙️  BASELINE TEST MODE: limit {batch_size} channels")

        youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
        bq_client = bigquery.Client(project=PROJECT_ID)
        now = datetime.utcnow()

        # Get channels to process
        if force:
            channel_query = f"""
                SELECT channel_id, channel_handle, vertical, tier
                FROM `{PROJECT_ID}.{DATASET_ID}.channels_to_track`
                ORDER BY RAND()
                LIMIT {batch_size}
            """
        else:
            channel_query = f"""
                SELECT ct.channel_id, ct.channel_handle, ct.vertical, ct.tier
                FROM `{PROJECT_ID}.{DATASET_ID}.channels_to_track` ct
                LEFT JOIN (
                    SELECT DISTINCT channel_id
                    FROM `{PROJECT_ID}.{DATASET_ID}.channel_baseline_videos`
                ) bl ON ct.channel_id = bl.channel_id
                WHERE bl.channel_id IS NULL
                ORDER BY RAND()
                LIMIT {batch_size}
            """

        try:
            channels = list(bq_client.query(channel_query).result())
        except Exception as e:
            print(f"❌ Baseline channel query failed: {e}")
            return f"BigQuery Query Error: {e}", 500

        if not channels:
            return "No channels need baseline polling", 200

        print(f"Baseline: processing {len(channels)} channels...")

        result = collect_channel_baselines(
            youtube, bq_client, channels, now, PROJECT_ID, DATASET_ID
        )

        if result["errors"]:
            return f"BQ Errors: {result['errors']}", 500

        msg = (
            f"Baseline complete: {result['rows_written']} videos across "
            f"{result['channels_done']} channels "
            f"({result['channels_skipped']} skipped) | "
            f"Quota: ~{result['quota_used']} units"
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