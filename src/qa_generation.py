
import re
import pandas as pd


def extract_findings(report):

    report = str(report)

    match = re.search(
        r"Findings:(.*?)(Impression:|$)",
        report,
        re.IGNORECASE | re.DOTALL
    )

    if match:
        return match.group(1).strip()

    return ""


def extract_impression(report):

    report = str(report)

    match = re.search(
        r"Impression:(.*)",
        report,
        re.IGNORECASE | re.DOTALL
    )

    if match:
        return match.group(1).strip()

    return ""


def yes_no_from_report(report, keywords):

    text = str(report).lower()

    if any(k.lower() in text for k in keywords):

        neg_patterns = []

        for k in keywords:
            neg_patterns.extend([
                rf"no[^.]*{re.escape(k.lower())}",
                rf"without[^.]*{re.escape(k.lower())}",
                rf"negative for[^.]*{re.escape(k.lower())}"
            ])

        for pattern in neg_patterns:
            if re.search(pattern, text):
                return "No"

        return "Yes"

    return "Not mentioned"


def create_qa_pairs(df, max_samples=100):

    qa_rows = []

    subset = df.head(max_samples).reset_index(drop=True)

    for idx, row in subset.iterrows():

        report = row["first_report"]

        findings = extract_findings(report)
        impression = extract_impression(report)

        # Findings QA
        qa_rows.append({
            "sample_id": idx,
            "subject_id": row["subject_id"],
            "image_path": row["image_path"],
            "question": "What are the findings in this chest X-ray?",
            "answer": findings if findings else report,
            "qa_type": "findings"
        })

        # Impression QA
        qa_rows.append({
            "sample_id": idx,
            "subject_id": row["subject_id"],
            "image_path": row["image_path"],
            "question": "What is the impression of this chest X-ray?",
            "answer": impression if impression else report,
            "qa_type": "impression"
        })

        # Pleural effusion
        qa_rows.append({
            "sample_id": idx,
            "subject_id": row["subject_id"],
            "image_path": row["image_path"],
            "question": "Is there pleural effusion?",
            "answer": yes_no_from_report(
                report,
                ["pleural effusion", "effusion"]
            ),
            "qa_type": "yes_no"
        })

        # Pneumothorax
        qa_rows.append({
            "sample_id": idx,
            "subject_id": row["subject_id"],
            "image_path": row["image_path"],
            "question": "Is there pneumothorax?",
            "answer": yes_no_from_report(
                report,
                ["pneumothorax"]
            ),
            "qa_type": "yes_no"
        })

        # Consolidation
        qa_rows.append({
            "sample_id": idx,
            "subject_id": row["subject_id"],
            "image_path": row["image_path"],
            "question": "Is there lung consolidation?",
            "answer": yes_no_from_report(
                report,
                ["consolidation"]
            ),
            "qa_type": "yes_no"
        })

    return pd.DataFrame(qa_rows)
