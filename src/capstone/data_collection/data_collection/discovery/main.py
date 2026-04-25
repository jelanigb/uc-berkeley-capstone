"""
Channel Fill-In — Cloud Run Function (v4)

Tier-aware targets: S/M/L per TARGETS config.
Short-circuits filled combos before hitting YouTube API.
Upload velocity filtering (>= 1 video/week).
Quota exceeded detection with 429 response.
Retries disabled (num_retries=0) to avoid wasting quota.
Per-run quota usage logging.

Candidate buffering (v4):
  Validated channels are written to channel_candidates table immediately
  upon passing velocity checks, before the combo loop completes. On each
  run, candidates are drained from the buffer first before making any
  search.list calls, so quota-crash mid-combo never loses work.

Test mode: append ?max=N to cap channels found per combo.
  e.g. ?max=1 finds at most 1 channel per combo
  Omit or ?max=0 for real targets.

Deploy:
  gcloud functions deploy channel-discovery \
    --gen2 --runtime python312 --region us-central1 \
    --source . --entry-point run_discovery --trigger-http \
    --allow-unauthenticated --memory 512Mi --timeout 240s \
    --set-secrets "YOUTUBE_API_KEY=YOUTUBE_API_KEY:latest" \
    --set-env-vars "PROJECT_ID=maduros-dolce"
"""

import os
import functions_framework
import io, json

from datetime import datetime, timedelta
from google.cloud.bigquery import LoadJobConfig, SourceFormat
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.cloud import bigquery



# --- Configuration ---
YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY")
PROJECT_ID = os.environ.get("PROJECT_ID", "maduros-dolce")
DATASET = "capstone_youtube"
TABLE_ID = f"{PROJECT_ID}.{DATASET}.channels_to_track"
CANDIDATES_TABLE = f"{PROJECT_ID}.{DATASET}.channel_candidates"

TARGETS = {"S": 150, "M": 100, "L": 75}

# It is difficult to find large channels which upload 1x / week. 
# 2x week should be sufficient.
MIN_VIDEOS_PER_WEEK = {"S": 1.0, "M": 1.0, "L": 0.5}
WEEKS_TO_CHECK = 4

FILL_QUERIES = {
    "Tech": [
        "tech review 2026", "coding tutorial beginner", "AI news",
        "linux tips", "app review", "gadget unboxing", "web development",
        "data science tutorial", "game development", "tech podcast",
        "smartphone comparison 2026", "best laptop 2026", "tech news today",
        "iPhone vs Samsung", "PC gaming setup", "software engineering career",
        "AI tools productivity", "Tesla tech review", "cybersecurity news",
        "mechanical keyboard review",
    ],
    "Lifestyle": [
        "daily vlog 2026", "home organization", "thrift haul",
        "workout routine", "healthy cooking", "apartment tour",
        "productivity tips", "self improvement", "slow living",
        "digital nomad", "skincare routine", "fashion haul 2026",
        "grocery haul", "couple vlog", "day in my life",
        "closet declutter", "home gym setup", "plant care",
        "morning routine", "budget travel",
    ],
    "Education": [
        "science explained", "math tutorial", "economics lesson",
        "history channel", "geography facts", "book summary",
        "study tips", "college advice", "online course review",
        "critical thinking", "psychology explained", "chemistry experiments",
        "world history", "financial literacy", "space documentary 2026",
        "biology explained", "debate analysis", "engineering explained",
        "philosophy lecture", "language learning tips",
    ],
}

TIER_RANGES = {
    "S": (1_000, 100_000),
    "M": (100_000, 1_000_000),
    "L": (1_000_000, 10_000_000),
}

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

def check_upload_velocity(youtube, channel_id, tier):
    """Returns (passes_filter: bool, velocity: float videos/week)."""
    playlist_id = "UU" + channel_id[2:]
    try:
        res = safe_youtube_call(
            youtube.playlistItems().list(
                playlistId=playlist_id, part="snippet", maxResults=50
            )
        )
    except YouTubeQuotaExceeded:
        raise
    except Exception:
        return False, 0.0

    items = res.get("items", [])
    if not items:
        return False, 0.0

    cutoff = datetime.utcnow() - timedelta(weeks=WEEKS_TO_CHECK)
    recent = sum(
        1 for i in items
        if datetime.strptime(
            i["snippet"]["publishedAt"], "%Y-%m-%dT%H:%M:%SZ"
        ) >= cutoff
    )
    vel = recent / WEEKS_TO_CHECK
    return vel >= MIN_VIDEOS_PER_WEEK[tier], round(vel, 2)


def get_current_counts(bq_client):
    """Returns dict like {('Tech','S'): 42, ...}"""
    query = f"""
        SELECT vertical, tier, COUNT(*) as cnt
        FROM `{TABLE_ID}`
        GROUP BY vertical, tier;
    """
    return {(r.vertical, r.tier): r.cnt for r in bq_client.query(query).result()}


def get_existing_ids(bq_client):
    """
    Returns set of all channel_ids already in tracking table OR
    candidate buffer (to avoid re-discovering the same channels).
    """
    tracked = {
        r.channel_id for r in bq_client.query(
            f"SELECT channel_id FROM `{TABLE_ID}`;"
        ).result()
    }
    buffered = {
        r.channel_id for r in bq_client.query(
            f"SELECT channel_id FROM `{CANDIDATES_TABLE}`;"
        ).result()
    }
    return tracked | buffered


def write_candidates(bq_client, rows):
    """
    Write newly validated channels to the candidate buffer immediately.
    Uses a load job (not streaming) so rows are immediately available
    for the UPDATE in drain_candidates().
    """
    if not rows:
        return
    load_config = LoadJobConfig(
        source_format=SourceFormat.NEWLINE_DELIMITED_JSON,
        write_disposition="WRITE_APPEND",
    )
    ndjson = "\n".join(json.dumps(r) for r in rows)
    load_job = bq_client.load_table_from_file(
        io.BytesIO(ndjson.encode("utf-8")),
        CANDIDATES_TABLE,
        job_config=load_config,
    )
    load_job.result()
    print(f"  💾 Buffered {len(rows)} candidate(s) to channel_candidates")

def drain_candidates(bq_client, vertical, tier, needed, existing_ids):
    """
    Pull pre-validated candidates from the buffer and insert them into
    channels_to_track. Returns the number of channels successfully drained.
    """
    query = f"""
        SELECT channel_id, channel_handle, vertical, tier,
               subscriber_count, total_channel_views,
               total_channel_videos, upload_velocity_per_week
        FROM `{CANDIDATES_TABLE}`
        WHERE vertical = '{vertical}'
          AND tier = '{tier}'
          AND added_to_tracking = FALSE
        LIMIT {needed};
    """
    rows = list(bq_client.query(query).result())
    if not rows:
        return 0

    tracking_rows = [
        {
            "channel_id": r.channel_id,
            "channel_handle": r.channel_handle,
            "vertical": r.vertical,
            "tier": r.tier,
            "subscriber_count": r.subscriber_count,
            "total_channel_views": r.total_channel_views,
            "total_channel_videos": r.total_channel_videos,
            "upload_velocity_per_week": r.upload_velocity_per_week,
        }
        for r in rows
    ]
    errors = bq_client.insert_rows_json(TABLE_ID, tracking_rows)
    if errors:
        print(f"  ❌ Failed to drain candidates for {vertical}/{tier}: {errors[:3]}")
        return 0

    # Mark as consumed
    ids = ", ".join(f"'{r.channel_id}'" for r in rows)
    bq_client.query(f"""
        UPDATE `{CANDIDATES_TABLE}`
        SET added_to_tracking = TRUE
        WHERE channel_id IN ({ids});
    """).result()

    for r in rows:
        existing_ids.add(r.channel_id)

    print(f"  📥 Drained {len(rows)} candidate(s) into channels_to_track "
          f"for {vertical}/{tier}")
    return len(rows)


# --- Core Logic ---

def fill_gaps(max_per_combo=0):
    """
    Finds underrepresented vertical+tier combos and fills them.

    On each run, drains the channel_candidates buffer first before making
    any search.list calls. Newly validated channels are written to the
    buffer immediately upon passing velocity checks, so a quota crash
    mid-combo never loses discovered work.

    Args:
        max_per_combo: If > 0, cap channels to find per combo (test mode).
                       If 0, use real targets.
    """
    if not YOUTUBE_API_KEY:
        raise ValueError("YOUTUBE_API_KEY environment variable not set")

    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    bq_client = bigquery.Client(project=PROJECT_ID)

    counts = get_current_counts(bq_client)
    existing_ids = get_existing_ids(bq_client)

    if max_per_combo > 0:
        print(f"⚙️  TEST MODE: max {max_per_combo} channel(s) per combo\n")

    # Build fill plan — skip full combos entirely
    print("=== Fill Plan ===")
    combos_to_fill = []
    combos_skipped = 0

    for tier in ["S", "M", "L"]:
        for vertical in FILL_QUERIES:
            target = TARGETS[tier]
            current = counts.get((vertical, tier), 0)
            needed = max(0, target - current)

            if needed == 0:
                print(f"  {vertical}/{tier}: {current}/{target} ✅ skip")
                combos_skipped += 1
            else:
                if max_per_combo > 0:
                    needed = min(needed, max_per_combo)
                print(f"  {vertical}/{tier}: {current}/{target} — need {needed}")
                combos_to_fill.append((vertical, tier, needed))

    # Quota forecast
    max_searches = sum(
        min(needed, len(FILL_QUERIES[v]))
        for v, _, needed in combos_to_fill
    )
    print(f"\n📊 Quota forecast:")
    print(f"   Combos to fill:    {len(combos_to_fill)}")
    print(f"   Combos skipped:    {combos_skipped}")
    print(f"   Max search calls:  {max_searches} × 100 = {max_searches * 100} units")
    print(f"   channels.list:     ~{max_searches} × 1 = ~{max_searches} units")
    print(f"   velocity checks:   ~{max_searches * 30} × 1 = ~{max_searches * 30} units (est.)")
    total_est = max_searches * 100 + max_searches + max_searches * 30
    print(f"   ─────────────────────────────────")
    print(f"   Estimated total:   ~{total_est} / 10,000 units\n")

    if not combos_to_fill:
        print("All combos are full. Nothing to do!")
        return "All combos full. 0 channels added."

    total_added = 0

    for vertical, tier, needed in combos_to_fill:
        print(f"🔎 Filling {vertical}/{tier} — need {needed}...")

        # --- Phase A: Drain candidate buffer first (free, no API calls) ---
        drained = drain_candidates(bq_client, vertical, tier, needed, existing_ids)
        needed -= drained
        total_added += drained

        if needed <= 0:
            print(f"  ✅ {vertical}/{tier} filled entirely from candidate buffer")
            continue

        # --- Phase B: Search YouTube for remaining needed ---
        newly_buffered = []

        for q in FILL_QUERIES[vertical]:
            if len(newly_buffered) >= needed:
                break

            response = safe_youtube_call(
                youtube.search().list(
                    part="snippet", type="channel", q=q,
                    maxResults=50, regionCode="US",
                )
            )

            ids = [
                item["id"]["channelId"]
                for item in response.get("items", [])
                if item["id"]["channelId"] not in existing_ids
            ]
            if not ids:
                continue

            stats_res = safe_youtube_call(
                youtube.channels().list(
                    part="statistics,snippet", id=",".join(ids[:50])
                )
            )

            lo, hi = TIER_RANGES[tier]
            batch_candidates = []

            for chan in stats_res.get("items", []):
                if len(newly_buffered) >= needed:
                    break
                cid = chan["id"]
                if cid in existing_ids:
                    continue

                subs = int(chan["statistics"].get("subscriberCount", 0))
                if not (lo <= subs < hi):
                    continue

                passes, vel = check_upload_velocity(youtube, cid, tier)
                if not passes:
                    print(f"    ✗ {chan['snippet'].get('customUrl', '?')} "
                          f"— {vel}/wk (too slow)")
                    continue

                existing_ids.add(cid)
                print(f"    ✓ {chan['snippet'].get('customUrl', '?')} "
                      f"— {subs:,} subs, {vel}/wk")

                candidate = {
                    "channel_id": cid,
                    "channel_handle": chan["snippet"].get("customUrl", "unknown"),
                    "vertical": vertical,
                    "tier": tier,
                    "subscriber_count": subs,
                    "total_channel_views": int(chan["statistics"].get("viewCount", 0)),
                    "total_channel_videos": int(chan["statistics"].get("videoCount", 0)),
                    "upload_velocity_per_week": vel,
                    "discovered_at": datetime.utcnow().isoformat(),
                    "added_to_tracking": False,
                }
                batch_candidates.append(candidate)
                newly_buffered.append(candidate)

            # Write each search batch to buffer immediately — before next API call
            if batch_candidates:
                write_candidates(bq_client, batch_candidates)

        # Drain whatever we just buffered into channels_to_track
        if newly_buffered:
            additional = drain_candidates(
                bq_client, vertical, tier, len(newly_buffered), existing_ids
            )
            total_added += additional
        else:
            print(f"  ⚠️  Couldn't find enough for {vertical}/{tier}")

    # Log quota usage
    quota_used = log_quota_usage()

    summary = (
        f"Fill complete. Added {total_added} channels "
        f"across {len(combos_to_fill)} combos. "
        f"Quota: ~{quota_used} units"
    )
    print(f"\n✅ {summary}")
    return summary


# --- Cloud Run Entry Point ---

@functions_framework.http
def run_discovery(request):
    try:
        max_per_combo = int(request.args.get("max", 0))
        result = fill_gaps(max_per_combo=max_per_combo)
        return result, 200

    except YouTubeQuotaExceeded as e:
        log_quota_usage()
        print(f"🚨 QUOTA EXCEEDED: {e}")
        return f"Quota exceeded: {e}", 429

    except Exception as e:
        log_quota_usage()
        print(f"❌ Unexpected error: {e}")
        return f"Server error: {e}", 500