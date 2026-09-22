from pathlib import Path
import re
import xml.etree.ElementTree as ET

import pandas as pd


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "temp_data"
PARLIAMENT_DIR = DATA_DIR / "parliament"
PARLAMINT_DIR = DATA_DIR / "parlamint"

TEI_DIR = PARLAMINT_DIR / "ParlaMint-DK.TEI"

OUTPUT_FILE = PARLAMINT_DIR / "parlamint_oda_labels.csv"


# --------------------------------------------------
# XML settings
# --------------------------------------------------

XML_ID = "{http://www.w3.org/XML/1998/namespace}id"

NS = {
    "tei": "http://www.tei-c.org/ns/1.0"
}

# --------------------------------------------------
# Extract agenda sections
# --------------------------------------------------

def extract_parlamint_agenda():

    rows = []

    xml_files = list(TEI_DIR.rglob("*.xml"))

    for xml_file in xml_files:

        tree = ET.parse(xml_file)
        root = tree.getroot()

        for div in root.findall(
            ".//tei:div[@type='debateSection']",
            NS
        ):

            head = div.find("tei:head", NS)
            note = div.find(
                "tei:note[@type='agendaItem']",
                NS
            )

            if head is None:
                continue

            domains = set()

            for u in div.findall(".//tei:u", NS):

                ana = u.attrib.get("ana", "")

                for token in ana.split():

                    if token.startswith("#domain."):

                        domains.add(
                            token.removeprefix("#domain.")
                        )

            agenda_id = (
                "".join(note.itertext()).strip()
                if note is not None
                else None
            )

            rows.append({
                "source_file": xml_file.name,
                "heading": "".join(
                    head.itertext()
                ).strip(),
                "agenda_id": agenda_id,
                "domains": sorted(domains)
            })

    return pd.DataFrame(rows)

# --------------------------------------------------
# Extract case number and session
# --------------------------------------------------

def extract_case_number(heading):

    if pd.isna(heading):
        return None

    match = re.search(
        r"\b(L|B|F|R)\s+(\d+)(?:\s*([A-Z]))?\b",
        heading
    )

    if not match:
        return None

    case_type = match.group(1)
    number = match.group(2)
    suffix = match.group(3)

    if suffix:
        return f"{case_type} {number} {suffix}"

    return f"{case_type} {number}"


def extract_session(filename):

    match = re.search(
        r"ParlaMint-DK_\d{4}-\d{2}-\d{2}-(\d+)-M\d+",
        filename
    )

    if match:
        return match.group(1)

    return None


def prepare_parlamint_agenda(df):

    df = df.copy()

    df["sag_nummer"] = (
        df["heading"].apply(extract_case_number)
    )

    df["session"] = (
        df["source_file"].apply(extract_session)
    )

    df["date"] = pd.to_datetime(
        df["agenda_id"].str[:10]
    )

    return df



# --------------------------------------------------
# Build only the consistent ParlaMint cases
# --------------------------------------------------

def make_consistent_parlamint_cases(df):

    domain_consistency = (
        df
        .dropna(subset=["sag_nummer"])
        .groupby(["session", "sag_nummer"])
        .agg(
            n_sections=("agenda_id", "size"),
            domain_sets=(
                "domains",
                lambda x: list(
                    {tuple(d) for d in x}
                )
            )
        )
        .reset_index()
    )

    domain_consistency["n_domain_sets"] = (
        domain_consistency["domain_sets"].apply(len)
    )

    consistent_keys = domain_consistency[
        domain_consistency["n_domain_sets"] == 1
    ][
        ["session", "sag_nummer"]
    ]

    df_cases = (
        df
        .merge(
            consistent_keys,
            on=["session", "sag_nummer"],
            how="inner"
        )
        .groupby(
            ["session", "sag_nummer"],
            as_index=False
        )
        .agg(
            first_date=("date", "min"),
            last_date=("date", "max"),
            heading=("heading", "first"),
            domains=("domains", "first")
        )
    )

    return df_cases


# --------------------------------------------------
# Link to ODA
# --------------------------------------------------

def load_oda_cases():

    file = PARLIAMENT_DIR / "roll_calls_resume.csv"

    df = pd.read_csv(file)

    return (
        df[
            [
                "sagid",
                "sag_nummer",
                "sag_titel",
                "sag_titelkort",
                "sag_resume",
                "dato"
            ]
        ]
        .drop_duplicates(subset="sagid")
        .reset_index(drop=True)
    )


def link_oda_to_parlamint(
    df_oda,
    df_parlamint
):

    df_oda = df_oda.copy()
    df_parlamint = df_parlamint.copy()

    # We match parliamentary activity by calendar date,
    # not by time of day.
    df_oda["dato"] = (
        pd.to_datetime(df_oda["dato"])
        .dt.normalize()
    )

    df_parlamint["first_date"] = (
        pd.to_datetime(df_parlamint["first_date"])
        .dt.normalize()
    )

    df_parlamint["last_date"] = (
        pd.to_datetime(df_parlamint["last_date"])
        .dt.normalize()
    )

    parlamint_start = df_parlamint["first_date"].min()
    parlamint_end = df_parlamint["last_date"].max()

    df_oda = df_oda[
        df_oda["dato"].between(
            parlamint_start,
            parlamint_end
        )
    ]

    candidates = df_oda.merge(
        df_parlamint,
        on="sag_nummer",
        how="inner"
    )

    matches = candidates[
        (candidates["dato"] >= candidates["first_date"])
        &
        (candidates["dato"] <= candidates["last_date"])
    ].copy()

    return matches



def main():

    print("Extracting ParlaMint agenda sections...")

    df_agenda = extract_parlamint_agenda()
    df_agenda = prepare_parlamint_agenda(
        df_agenda
    )

    print(
        f"Agenda sections: {len(df_agenda)}"
    )

    df_parlamint_cases = (
        make_consistent_parlamint_cases(
            df_agenda
        )
    )

    print(
        f"Consistent ParlaMint cases: "
        f"{len(df_parlamint_cases)}"
    )

    df_oda = load_oda_cases()

    df_labels = link_oda_to_parlamint(
        df_oda,
        df_parlamint_cases
    )

    # Keep only what we actually need downstream
    df_labels = df_labels[
        [
            "sagid",
            "sag_nummer",
            "domains"
        ]
    ].copy()

    # Store multi-label domains in a CSV-friendly format
    df_labels["domains"] = (
        df_labels["domains"]
        .apply(lambda domains: "|".join(domains))
    )

    df_labels.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"Linked ODA cases: {len(df_labels)}"
    )

    print(
        f"Saved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()

