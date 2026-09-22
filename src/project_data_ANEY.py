import re
from pathlib import Path
import pandas as pd

# --------------------------------------------------
# Paths
# --------------------------------------------------

DATA_DIR = Path("../data/temp_data")

CANDIDATE_DIR = DATA_DIR / "candidate_test"
PARLIAMENT_DIR = DATA_DIR / "parliament"
CAP_DIR = DATA_DIR / "cap"
PARLAMINT_DIR = DATA_DIR / "parlamint"

# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def normalize_title(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text

# --------------------------------------------------
# Candidate-test data
# --------------------------------------------------

def load_candidate_tests():

    file = CANDIDATE_DIR / "Kandidattestdata.xlsx"

    elections = ["FV11", "FV15", "FV19", "FV22"]

    candidate_tests = {}

    for election in elections:
        candidate_tests[election] = pd.read_excel(
            file,
            sheet_name=election
        )

    return candidate_tests


def make_candidate_questions(candidate_tests):

    frames = []

    for election, df in candidate_tests.items():

        questions = (
            df[["Question"]]
            .drop_duplicates()
            .dropna()
            .reset_index(drop=True)
        )

        questions["election"] = election

        frames.append(questions)

    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------
# Parliament data
# --------------------------------------------------

def load_roll_calls():

    return pd.read_csv(
        PARLIAMENT_DIR / "roll_calls_resume.csv"
    )


def make_cases(df_roll_calls):

    return (
        df_roll_calls[
            [
                "sagid",
                "sag_nummer",
                "sag_titel",
                "sag_titelkort",
                "sag_resume",
                "dato",
            ]
        ]
        .drop_duplicates(subset="sagid")
        .reset_index(drop=True)
    )


def load_votes():

    return pd.read_csv(
        PARLIAMENT_DIR / "stemme.csv"
    )


def load_politicians():

    return pd.read_csv(
        PARLIAMENT_DIR / "politicians.csv"
    )


def load_actor_candidate_map():

    return pd.read_csv(
        PARLIAMENT_DIR / "actor_candidate_map.csv"
    )



def load_oda_case_topics():
    file = PARLIAMENT_DIR / "oda_case_topics.csv"
    df = pd.read_csv(file)

    df["oda_sagsomraade"] = (
        df["oda_sagsomraade"]
        .fillna("")
        .apply(lambda x: x.split("|") if x else [])
    )

    return df

# --------------------------------------------------
# CAP data
# --------------------------------------------------

def load_cap_bills():

    return pd.read_csv(
        CAP_DIR / "Love_01092019_-_Sheet1.csv"
    )


def make_cap_labels(df_cases, df_cap_bills):
    cases = df_cases.copy()
    cap_bills = df_cap_bills.copy()

    # Normalize titles
    cases["title_normalized"] = (
        cases["sag_titel"]
        .apply(normalize_title)
    )

    cap_bills["title_normalized"] = (
        cap_bills["description"]
        .apply(normalize_title)
    )

    # Calendar year
    cases["calendar_year"] = (
        pd.to_datetime(cases["dato"])
        .dt.year
        .astype("Int64")
    )

    cap_bills["calendar_year"] = (
        pd.to_numeric(
            cap_bills["calendar_year"],
            errors="coerce"
        )
        .astype("Int64")
    )

    # Match ODA cases to CAP bills
    matched = cases.merge(
        cap_bills[
            [
                "title_normalized",
                "calendar_year",
                "majortopic",
                "subtopic"
            ]
        ],
        on=["title_normalized", "calendar_year"],
        how="inner"
    )

    # Check that duplicate matches do not disagree
    label_agreement = (
        matched
        .groupby("sagid")
        .agg(
            n_major_topics=("majortopic", "nunique"),
            n_subtopics=("subtopic", "nunique")
        )
    )

    conflicts = label_agreement[
        (label_agreement["n_major_topics"] > 1)
        | (label_agreement["n_subtopics"] > 1)
    ]

    if not conflicts.empty:
        raise ValueError(
            f"Found {len(conflicts)} ODA cases with conflicting CAP labels."
        )

    # One CAP label per ODA case
    cap_labels = (
        matched[
            [
                "sagid",
                "sag_nummer",
                "sag_titel",
                "dato",
                "majortopic",
                "subtopic"
            ]
        ]
        .drop_duplicates(subset="sagid")
        .reset_index(drop=True)
    )

    return cap_labels

# --------------------------------------------------
# ParlaMint data
# --------------------------------------------------

def load_parlamint_labels():
    file = PARLAMINT_DIR / "parlamint_oda_labels.csv"

    df = pd.read_csv(file)

    df["domains"] = (
        df["domains"]
        .fillna("")
        .apply(lambda x: x.split("|") if x else [])
    )

    return df

# --------------------------------------------------
# Load everything
# --------------------------------------------------

def load_project_data():
    candidate_tests = load_candidate_tests()
    df_questions = make_candidate_questions(candidate_tests)

    df_roll_calls = load_roll_calls()
    df_cases = make_cases(df_roll_calls)

    df_cap_bills = load_cap_bills()

    df_cap_labels = make_cap_labels(
        df_cases,
        df_cap_bills
    )

    df_parlamint_labels = load_parlamint_labels()

    return {
        "candidate_tests": candidate_tests,
        "questions": df_questions,
        "roll_calls": df_roll_calls,
        "cases": df_cases,
        "votes": load_votes(),
        "politicians": load_politicians(),
        "actor_candidate_map": load_actor_candidate_map(),
        "cap_bills": df_cap_bills,
        "cap_labels": df_cap_labels,
        "parlamint_labels": df_parlamint_labels,
        "df_oda_topics": load_oda_case_topics(),
    }