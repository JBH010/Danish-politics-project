"""Fetch VoteWatch EP7/EP8 data for Danish MEPs into period folders."""
from __future__ import annotations

import zipfile
import urllib.request
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
CACHE_DIR = DATA_DIR / "_votewatch_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

VW_ZIP_URL = (
    "https://cadmus.eui.eu/bitstreams/d564c696-e6d7-551d-971f-abad4447525e/download"
)
VW_ZIP_PATH = CACHE_DIR / "VoteWatch-EP-voting-data_2004-2022.zip"

EP_TERMS_VW = {
    7: {
        "folder": "2009-2014",
        "rcv": "EP7_RCVs_2014_06_19.xlsx",
        "docs": "EP7_Voted docs.xlsx",
    },
    8: {
        "folder": "2014-2019",
        "rcv": "EP8_RCVs_2019_06_25.xlsx",
        "docs": "EP8_Voted docs.xlsx",
    },
}

POSITION_MAP = {
    1: "FOR",
    2: "AGAINST",
    3: "ABSTENTION",
    4: "DID_NOT_VOTE",
    0: "DID_NOT_VOTE",
}


def download_votewatch_zip() -> Path:
    if VW_ZIP_PATH.exists():
        return VW_ZIP_PATH
    print("Downloading VoteWatch archive (~63 MB) from Cadmus EUI...")
    req = urllib.request.Request(VW_ZIP_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        VW_ZIP_PATH.write_bytes(resp.read())
    print("Download complete.")
    return VW_ZIP_PATH


def _meta_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if not isinstance(c, int)]


def _vote_columns(df: pd.DataFrame) -> list[int]:
    return [c for c in df.columns if isinstance(c, int)]


def _member_id_series(df: pd.DataFrame) -> pd.Series:
    if "WebisteEpID" in df.columns:
        return pd.to_numeric(df["WebisteEpID"], errors="coerce").astype("Int64")
    return pd.to_numeric(df["MEP ID"], errors="coerce").astype("Int64")


def process_votewatch_term(term: int) -> None:
    cfg = EP_TERMS_VW[term]
    out_dir = DATA_DIR / cfg["folder"]
    out_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(download_votewatch_zip()) as zf:
        rcv = pd.read_excel(zf.open(cfg["rcv"]))
        docs = pd.read_excel(zf.open(cfg["docs"]))

    dk = rcv[rcv["Country"] == "Denmark"].copy()
    meta_cols = _meta_columns(dk)
    vote_cols = _vote_columns(dk)

    members = pd.DataFrame(
        {
            "id": _member_id_series(dk),
            "votewatch_mep_id": pd.to_numeric(dk.get("MEP ID"), errors="coerce"),
            "first_name": dk["Fname"],
            "last_name": dk["Lname"],
            "country_code": "DNK",
            "national_party": dk["Party"],
            "ep_group": dk["EPG"],
            "source": "votewatch",
        }
    )
    members.to_csv(out_dir / "ep_members_dk.csv", index=False)

    long = dk.melt(
        id_vars=meta_cols,
        value_vars=vote_cols,
        var_name="vote_id",
        value_name="vote_code",
    )
    long["vote_code"] = pd.to_numeric(long["vote_code"], errors="coerce")
    long = long[long["vote_code"].notna() & (long["vote_code"] != 5)]
    long["position"] = long["vote_code"].map(POSITION_MAP).fillna("DID_NOT_VOTE")

    member_votes = pd.DataFrame(
        {
            "vote_id": long["vote_id"].astype(int),
            "member_id": _member_id_series(long),
            "position": long["position"],
            "country_code": "DNK",
            "ep_group": long["EPG"],
            "source": "votewatch",
        }
    )

    votes = pd.DataFrame(
        {
            "id": pd.to_numeric(docs["Vote ID"], errors="coerce"),
            "timestamp": pd.to_datetime(docs["Date"], errors="coerce", utc=True),
            "display_title": docs["Title"],
            "procedure_reference": docs.get("Procedure"),
            "procedure_type": docs.get("Leg/Non-Leg/Bud"),
            "vote_type": docs.get("Type of Vote"),
            "voting_rule": docs.get("Voting Rule"),
            "source": "votewatch",
        }
    )
    if "Vote Yeas" in docs.columns:
        votes["count_for"] = pd.to_numeric(
            docs["Vote Yeas"].astype(str).str.replace(r"[^\d]", "", regex=True),
            errors="coerce",
        )
        votes["count_against"] = pd.to_numeric(docs["No"], errors="coerce")
        votes["count_abstention"] = pd.to_numeric(docs["Abs"], errors="coerce")

    votes = votes.dropna(subset=["id"])
    votes["id"] = votes["id"].astype(int)

    memberships = pd.DataFrame(
        {
            "member_id": _member_id_series(dk),
            "votewatch_mep_id": pd.to_numeric(dk.get("MEP ID"), errors="coerce"),
            "term": term,
            "ep_group": dk["EPG"],
            "national_party": dk["Party"],
            "start_date": pd.to_datetime(dk.get("Start"), errors="coerce").dt.date,
            "end_date": pd.to_datetime(dk.get("End"), errors="coerce").dt.date,
            "source": "votewatch",
        }
    )
    memberships.to_csv(out_dir / "ep_group_memberships_dk.csv", index=False)

    # keep only votes that DK MEPs participated in
    active_vote_ids = set(member_votes["vote_id"].astype(int))
    votes_filtered = votes[votes["id"].isin(active_vote_ids)]
    member_votes = member_votes[member_votes["vote_id"].isin(active_vote_ids)]
    votes_filtered.to_csv(out_dir / "ep_votes.csv", index=False)
    member_votes.to_csv(out_dir / "ep_member_votes_dk.csv", index=False)

    print(
        f"{cfg['folder']}: {len(members)} MEPs, "
        f"{len(votes_filtered)} votes, {len(member_votes)} member-vote rows"
    )


if __name__ == "__main__":
    for term in sorted(EP_TERMS_VW):
        process_votewatch_term(term)
