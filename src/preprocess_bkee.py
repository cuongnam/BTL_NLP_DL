"""
preprocess_bkee.py

Preprocessing pipeline for the BKEE Vietnamese Event Extraction dataset.

This script prepares datasets for:

1. Trigger Detection
2. Event Type Classification
3. Argument Extraction

Author: Nguyễn Nam Cường
"""

from pathlib import Path
from collections import Counter, defaultdict
import json
import logging
from typing import Dict, List, Any

# ============================================================
# CONFIGURATION
# ============================================================

# ROOT_DIR = Path(__file__).resolve().parent
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "processed"

OUTPUT_DIR = ROOT_DIR / "data" / "preprocessed"

TRIGGER_DIR = OUTPUT_DIR / "trigger"
EVENT_DIR = OUTPUT_DIR / "event_type"
ARGUMENT_DIR = OUTPUT_DIR / "argument"

LABEL_MAP_PATH = OUTPUT_DIR / "label_maps.json"
STATISTICS_PATH = OUTPUT_DIR / "statistics.json"

SPLITS = ["train", "dev", "test"]

# ============================================================
# LOGGER
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)

# ============================================================
# UTILITIES
# ============================================================


def ensure_directories() -> None:
    """
    Create output folders if they do not exist.
    """

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TRIGGER_DIR.mkdir(parents=True, exist_ok=True)
    EVENT_DIR.mkdir(parents=True, exist_ok=True)
    ARGUMENT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DATA LOADING
# ============================================================


def load_dataset(path: Path) -> List[Dict]:
    """
    Load a jsonl dataset.

    Each line is one json object.

    Parameters
    ----------
    path : Path

    Returns
    -------
    List[Dict]
    """

    logger.info(f"Loading {path.name}")

    data = []

    with open(path, "r", encoding="utf8") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            record = json.loads(line)

            data.append(record)

    logger.info(f"Loaded {len(data)} sentences")

    return data


# ============================================================
# DATA VALIDATION
# ============================================================


def validate_record(record: Dict) -> bool:
    """
    Validate one record.

    Returns
    -------
    bool
    """

    required_fields = [
        "doc_id",
        "tokens",
        "sentence",
        "entity_mentions",
        "event_mentions"
    ]

    for field in required_fields:

        if field not in record:

            logger.warning(
                f"{record.get('doc_id','UNKNOWN')} "
                f"missing field: {field}"
            )

            return False

    tokens = record["tokens"]

    token_num = len(tokens)

    # validate entity span

    for entity in record["entity_mentions"]:

        if entity["start"] >= token_num:

            logger.warning(
                f"Invalid entity start index "
                f"in {record['doc_id']}"
            )

            return False

        if entity["end"] > token_num:

            logger.warning(
                f"Invalid entity end index "
                f"in {record['doc_id']}"
            )

            return False

    # validate trigger span

    for event in record["event_mentions"]:

        trigger = event["trigger"]

        if trigger["start"] >= token_num:

            logger.warning(
                f"Invalid trigger start "
                f"in {record['doc_id']}"
            )

            return False

        if trigger["end"] > token_num:

            logger.warning(
                f"Invalid trigger end "
                f"in {record['doc_id']}"
            )

            return False

    return True


def validate_dataset(records: List[Dict]) -> List[Dict]:
    """
    Remove invalid samples.

    Returns
    -------
    List[Dict]
    """

    valid_records = []

    invalid = 0

    for record in records:

        if validate_record(record):

            valid_records.append(record)

        else:

            invalid += 1

    logger.info(f"Valid samples : {len(valid_records)}")
    logger.info(f"Invalid samples : {invalid}")

    return valid_records


# ============================================================
# DATASET STATISTICS
# ============================================================


def compute_statistics(records: List[Dict]) -> Dict:
    """
    Compute dataset statistics.

    Returns
    -------
    Dict
    """

    stats = {}

    event_counter = Counter()

    entity_counter = Counter()

    role_counter = Counter()

    total_events = 0

    total_entities = 0

    for record in records:

        total_entities += len(record["entity_mentions"])

        total_events += len(record["event_mentions"])

        for entity in record["entity_mentions"]:

            entity_counter[entity["entity_type"]] += 1

        for event in record["event_mentions"]:

            event_counter[event["event_type"]] += 1

            for arg in event["arguments"]:

                role_counter[arg["role"]] += 1

    stats["sentences"] = len(records)

    stats["events"] = total_events

    stats["entities"] = total_entities

    stats["event_distribution"] = dict(event_counter)

    stats["entity_distribution"] = dict(entity_counter)

    stats["argument_distribution"] = dict(role_counter)

    return stats


# ============================================================
# SAVE JSON
# ============================================================


def save_json(data: Any, path: Path) -> None:
    """
    Save json with UTF-8 encoding.
    """

    with open(path, "w", encoding="utf8") as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2
        )

    logger.info(f"Saved -> {path}")
# ============================================================
# TRIGGER LABELS
# ============================================================


TRIGGER_O = "O"
TRIGGER_B = "B-TRIGGER"
TRIGGER_I = "I-TRIGGER"


def create_trigger_labels(record: Dict) -> List[str]:
    """
    Convert trigger spans into BIO labels.

    Parameters
    ----------
    record

    Returns
    -------
    List[str]
    """

    tokens = record["tokens"]

    labels = [TRIGGER_O] * len(tokens)

    for event in record["event_mentions"]:

        trigger = event["trigger"]

        start = trigger["start"]
        end = trigger["end"]

        if start >= len(tokens):
            continue

        labels[start] = TRIGGER_B

        for idx in range(start + 1, end):
            labels[idx] = TRIGGER_I

    return labels


# ============================================================
# BUILD TRIGGER DATASET
# ============================================================


def build_trigger_dataset(records: List[Dict]) -> List[Dict]:
    """
    Build dataset for Trigger Detection.

    Returns
    -------
    List[Dict]
    """

    dataset = []

    for record in records:

        sample = {

            "id": record["doc_id"],

            "sentence": record["sentence"],

            "tokens": record["tokens"],

            "pieces": record["pieces"],

            "token_lens": record["token_lens"],

            "trigger_labels": create_trigger_labels(record)

        }

        dataset.append(sample)

    logger.info(
        f"Trigger dataset created : {len(dataset)} samples"
    )

    return dataset


# ============================================================
# TRIGGER LABEL MAP
# ============================================================


def build_trigger_label_map():
    """
    Fixed label map for trigger detection.
    """

    label2id = {

        "O": 0,

        "B-TRIGGER": 1,

        "I-TRIGGER": 2

    }

    id2label = {

        0: "O",

        1: "B-TRIGGER",

        2: "I-TRIGGER"

    }

    return label2id, id2label


# ============================================================
# SAVE TRIGGER DATASET
# ============================================================


def save_trigger_dataset(dataset: List[Dict], split: str):

    path = TRIGGER_DIR / f"{split}.json"

    save_json(dataset, path)
# ============================================================
# EVENT TYPE DATASET
# ============================================================


def build_event_dataset(records: List[Dict]) -> List[Dict]:
    """
    Build dataset for Event Type Classification.

    One event = one training sample.
    """

    dataset = []

    for record in records:

        for event in record["event_mentions"]:

            trigger = event["trigger"]

            sample = {

                "id": event["id"],

                "doc_id": record["doc_id"],

                "sentence": record["sentence"],

                "tokens": record["tokens"],

                "pieces": record["pieces"],

                "token_lens": record["token_lens"],

                "trigger": {

                    "text": trigger["text"],

                    "start": trigger["start"],

                    "end": trigger["end"]

                },

                "event_type": event["event_type"]

            }

            dataset.append(sample)

    logger.info(
        f"Event dataset created : {len(dataset)} samples"
    )

    return dataset


# ============================================================
# ARGUMENT LABELS
# ============================================================


def create_argument_labels(record: Dict,
                           event: Dict) -> List[str]:
    """
    Create BIO labels for one event.

    One event = one label sequence.
    """

    labels = ["O"] * len(record["tokens"])

    entity_map = {}

    for entity in record["entity_mentions"]:

        entity_map[entity["id"]] = entity

    for argument in event["arguments"]:

        entity = entity_map.get(argument["entity_id"])

        if entity is None:
            continue

        role = argument["role"]

        start = entity["start"]

        end = entity["end"]

        labels[start] = f"B-{role}"

        for idx in range(start + 1, end):

            labels[idx] = f"I-{role}"

    return labels


# ============================================================
# ARGUMENT DATASET
# ============================================================


# def build_argument_dataset(records: List[Dict]) -> List[Dict]:
#     """
#     One event = one sample.
#     """

#     dataset = []

#     for record in records:

#         for event in record["event_mentions"]:

#             trigger = event["trigger"]

#             sample = {

#                 "id": event["id"],

#                 "doc_id": record["doc_id"],

#                 "sentence": record["sentence"],

#                 "tokens": record["tokens"],

#                 "pieces": record["pieces"],

#                 "token_lens": record["token_lens"],

#                 "trigger": {

#                     "text": trigger["text"],

#                     "start": trigger["start"],

#                     "end": trigger["end"]

#                 },

#                 "event_type": event["event_type"],

#                 "argument_labels":
#                     create_argument_labels(
#                         record,
#                         event
#                     )

#             }

#             dataset.append(sample)

#     logger.info(
#         f"Argument dataset created : {len(dataset)} samples"
#     )

#     return dataset

def build_argument_dataset(records: List[Dict]) -> List[Dict]:
    """
    One event = one sample.
    Đã cải tiến: Chèn thêm marker <tg> và </tg> bọc quanh Trigger trong mảng pieces 
    và cập nhật lại token_lens tương ứng để mồi thông tin ngữ cảnh cho PhoBERT.
    """

    dataset = []

    for record in records:

        for event in record["event_mentions"]:

            trigger = event["trigger"]
            trigger_start = trigger["start"]
            trigger_end = trigger["end"]

            # --- LOGIC CẢI TIẾN: CHÈN MARKER ĐÁNH DẤU TRIGGER ---
            new_pieces = []
            new_token_lens = []
            current_piece_idx = 0

            for word_idx, length in enumerate(record["token_lens"]):
                # Nếu bắt đầu đến từ Trigger, chèn thêm token mở <tg>
                if word_idx == trigger_start:
                    new_pieces.append("<tg>")
                    new_token_lens.append(1) # Token đặc biệt tính độ dài là 1

                # Lấy các subwords (pieces) thuộc về từ hiện tại
                word_pieces = record["pieces"][current_piece_idx : current_piece_idx + length]
                new_pieces.extend(word_pieces)
                new_token_lens.append(length)
                current_piece_idx += length

                # Nếu kết thúc từ Trigger, chèn thêm token đóng </tg>
                if word_idx == trigger_end - 1:
                    new_pieces.append("</tg>")
                    new_token_lens.append(1) # Token đặc biệt tính độ dài là 1
            # ----------------------------------------------------

            sample = {

                "id": event["id"],

                "doc_id": record["doc_id"],

                "sentence": record["sentence"],

                "tokens": record["tokens"],

                # Sử dụng mảng pieces và token_lens mới đã có Marker
                "pieces": new_pieces,

                "token_lens": new_token_lens,

                "trigger": {

                    "text": trigger["text"],

                    "start": trigger["start"],

                    "end": trigger["end"]

                },

                "event_type": event["event_type"],

                "argument_labels":
                    create_argument_labels(
                        record,
                        event
                    )

            }

            dataset.append(sample)

    logger.info(
        f"Argument dataset created (with Trigger Markers): {len(dataset)} samples"
    )

    return dataset

# ============================================================
# LABEL MAPS
# ============================================================


def build_label_maps(records: List[Dict]):
    """
    Automatically build all label maps.
    """

    event_types = set()

    argument_roles = set()

    for record in records:

        for event in record["event_mentions"]:

            event_types.add(event["event_type"])

            for arg in event["arguments"]:

                argument_roles.add(arg["role"])

    # ---------- Event Type ----------

    event_type2id = {}

    id2event_type = {}

    for idx, label in enumerate(sorted(event_types)):

        event_type2id[label] = idx

        id2event_type[idx] = label

    # ---------- Argument ----------

    argument_label2id = {"O": 0}

    argument_id2label = {0: "O"}

    current = 1

    for role in sorted(argument_roles):

        argument_label2id[f"B-{role}"] = current
        argument_id2label[current] = f"B-{role}"
        current += 1

        argument_label2id[f"I-{role}"] = current
        argument_id2label[current] = f"I-{role}"
        current += 1

    return {

        "event_type": {

            "label2id": event_type2id,

            "id2label": id2event_type

        },

        "argument": {

            "label2id": argument_label2id,

            "id2label": argument_id2label

        }

    }


# ============================================================
# SAVE DATASETS
# ============================================================


def save_event_dataset(dataset, split):

    save_json(
        dataset,
        EVENT_DIR / f"{split}.json"
    )


def save_argument_dataset(dataset, split):

    save_json(
        dataset,
        ARGUMENT_DIR / f"{split}.json"
    )
# ============================================================
# EXPORT LABEL MAPS
# ============================================================


def export_label_maps(all_records: List[Dict]) -> None:
    """
    Export all label maps.

    Parameters
    ----------
    all_records : train + dev + test
    """

    trigger_label2id, trigger_id2label = build_trigger_label_map()

    other_maps = build_label_maps(all_records)

    label_maps = {

        "trigger": {

            "label2id": trigger_label2id,

            "id2label": trigger_id2label

        },

        "event_type": other_maps["event_type"],

        "argument": other_maps["argument"]

    }

    save_json(label_maps, LABEL_MAP_PATH)


# ============================================================
# PROCESS ONE SPLIT
# ============================================================


def process_split(split: str):
    """
    Process one dataset split.

    Returns
    -------
    records
    statistics
    """

    logger.info("=" * 60)
    logger.info(f"Processing {split}")

    dataset_path = DATA_DIR / f"{split}.json"

    records = load_dataset(dataset_path)

    records = validate_dataset(records)

    statistics = compute_statistics(records)

    trigger_dataset = build_trigger_dataset(records)

    event_dataset = build_event_dataset(records)

    argument_dataset = build_argument_dataset(records)

    save_trigger_dataset(trigger_dataset, split)

    save_event_dataset(event_dataset, split)

    save_argument_dataset(argument_dataset, split)

    return records, statistics


# ============================================================
# MAIN
# ============================================================


def main():

    logger.info("BKEE PREPROCESSING")

    ensure_directories()

    statistics = {}

    all_records = []

    for split in SPLITS:

        records, stats = process_split(split)

        statistics[split] = stats

        all_records.extend(records)

    export_label_maps(all_records)

    save_json(statistics, STATISTICS_PATH)

    logger.info("=" * 60)
    logger.info("Finished preprocessing.")
    logger.info(f"Output folder : {OUTPUT_DIR}")


if __name__ == "__main__":

    main()