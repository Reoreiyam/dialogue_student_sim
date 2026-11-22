"""Prepare the OSF Tutoring Chat Log Dataset for sim_student.

The source dataset stores messages, session metadata, and problem metadata in
separate CSV files.  This script joins them into the repository's canonical
dialogue format and creates deterministic student-level train/val/test splits.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("wh5n9-osfstorage-archive/Tutoring Chat Log Dataset"),
    )
    parser.add_argument("--output", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, default=221)
    return parser.parse_args()


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).replace("\r", " ").replace("\n", " ").strip()


def build_dialogues(source: Path) -> pd.DataFrame:
    messages = pd.read_csv(source / "chat_message_logs.csv")
    requests = pd.read_csv(source / "chat_request_logs.csv")
    problems = pd.read_csv(source / "problem_details.csv")
    sequences = pd.read_csv(source / "sequence_details.csv")

    messages["user"] = messages["user"].str.upper()
    messages["timestamp"] = pd.to_numeric(messages["timestamp"], errors="coerce")
    messages = messages.sort_values(["session_id", "timestamp"])

    sessions = messages.groupby("session_id", sort=False)
    request_by_session = requests.drop_duplicates("session_id").set_index("session_id")
    problem_by_id = problems.drop_duplicates("problem_id").set_index("problem_id")
    sequence_by_id = sequences.drop_duplicates("sequence_id").set_index("sequence_id")

    records = []
    for session_id, group in sessions:
        if session_id not in request_by_session.index:
            continue
        request = request_by_session.loc[session_id]
        problem_id = request.get("problem_id")
        problem = problem_by_id.loc[problem_id] if problem_id in problem_by_id.index else None
        sequence_id = request.get("sequence_id")
        sequence = sequence_by_id.loc[sequence_id] if sequence_id in sequence_by_id.index else None

        turns = []
        for _, row in group.iterrows():
            user = clean_text(row["user"])
            role = "tutor" if user == "TUTOR" else "student"
            content = clean_text(row["message"])
            if not content:
                continue
            if turns and turns[-1]["role"] == role:
                turns[-1]["content"] += " " + content
            else:
                turns.append({"role": role, "content": content})
        if not turns or not any(turn["role"] == "student" for turn in turns):
            continue

        question = clean_text(problem["problem_html"]) if problem is not None else ""
        skill_code = clean_text(problem["problem_skill_code"]) if problem is not None else ""
        skill_description = clean_text(problem["problem_skill_description"]) if problem is not None else ""
        subject = skill_code or skill_description
        sequence_name = clean_text(sequence["sequence_name"]) if sequence is not None else ""

        records.append(
            {
                "key": str(session_id),
                "session_id": str(session_id),
                "studentID": request.get("student_id"),
                "tutorID": request.get("tutor_id"),
                "QuestionID": problem_id,
                "InterventionId": request.get("assignment_id"),
                "sequence_id": sequence_id,
                "question": question,
                "subjects": json.dumps([subject] if subject else [], ensure_ascii=False),
                "sequence_name": sequence_name,
                "turns": json.dumps(turns, ensure_ascii=False),
            }
        )

    return pd.DataFrame(records)


def main() -> None:
    args = parse_args()
    dialogues = build_dialogues(args.source)
    if dialogues.empty:
        raise RuntimeError("No dialogue with a student turn was found")

    # Split by student, preventing the same student from appearing in multiple splits.
    students = list(dialogues["studentID"].dropna().unique())
    random.Random(args.seed).shuffle(students)
    n_train = int(len(students) * 0.8)
    n_val = int(len(students) * 0.1)
    train_students = set(students[:n_train])
    val_students = set(students[n_train : n_train + n_val])

    split_names = {
        "train": train_students,
        "val": val_students,
        "test": set(students) - train_students - val_students,
    }
    out_dir = args.output / "annotated" / "tutoring_chat"
    out_dir.mkdir(parents=True, exist_ok=True)
    (args.output / "overgenerated" / "tutoring_chat").mkdir(parents=True, exist_ok=True)
    (args.output / "raw" / "tutoring_chat").mkdir(parents=True, exist_ok=True)

    for split, split_students in split_names.items():
        part = dialogues[dialogues["studentID"].isin(split_students)]
        part.to_csv(out_dir / f"{split}.csv", index=False)

    manifest = {
        "source": str(args.source),
        "dataset": "tutoring_chat",
        "seed": args.seed,
        "dialogues": len(dialogues),
        "students": len(students),
        "splits": {split: int(sum(dialogues["studentID"].isin(ids))) for split, ids in split_names.items()},
    }
    (args.output / "tutoring_chat_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
