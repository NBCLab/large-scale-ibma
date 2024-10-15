"""Get NeuroVault Collections linked to PubMed articles."""

import argparse
import os.path as op
import re
import urllib.parse

import numpy as np
import pandas as pd
import requests

NEUROSCOUT_OWNER_ID = 5761


def _get_parser():
    parser = argparse.ArgumentParser(description="Download NeuroVault data")
    parser.add_argument(
        "--project_dir",
        dest="project_dir",
        required=True,
        help="Path to project directory",
    )
    parser.add_argument(
        "--neurovault_version",
        dest="neurovault_version",
        required=False,
        default="february_2024",
        help="NeuroVault version",
    )
    parser.add_argument(
        "--pg_query_id",
        dest="pg_query_id",
        required=False,
        default="a444c1d1cc79f746a519d97ce9672089",
        help="Pubget query ID",
    )
    return parser


def get_pmid_from_doi(doi):
    """Query PubMed for the PMID of a paper based on its DOI."""
    url = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&"
        f'term="{doi}"&retmode=json'
    )
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"PubMed API returned status code {response.status_code} for {url}")
    data = response.json()
    if data["esearchresult"]["idlist"]:
        return data["esearchresult"]["idlist"][0]
    else:
        return None


def get_pmcid_from_pmid(pmid):
    """Query PubMed for the PMC ID of a paper based on its PMID."""
    url = f"https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/?ids={pmid}&format=json"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"PubMed API returned status code {response.status_code} for {url}")
    data = response.json()
    if data["records"] and "pmcid" in data["records"][0]:
        pmcid = data["records"][0]["pmcid"]
        return pmcid[3:] if pmcid.startswith("PMC") else pmcid
    else:
        return None


def get_pmid_pmcid_from_doi(doi):
    pmid = get_pmid_from_doi(doi)
    if pmid is None:
        return pmid, None

    pmcid = get_pmcid_from_pmid(pmid)

    return pmid, pmcid


def _check_string(s):
    return all(c.isdigit() for c in s)


def _convert_collection_id(collection_id, collections_df):
    if str(collection_id).isalpha():
        matches = collections_df[collections_df.private_token == collection_id]
        return matches.id.values[0] if matches.size > 0 else None
    else:
        return int(collection_id) if _check_string(str(collection_id)) else None


def _look_up_doi(row):
    doi_regex = re.compile(r"10.\d{4,9}/[-._;()/:a-zA-Z0-9]+")

    if isinstance(row.description, str):
        dois = re.findall(doi_regex, row.description)
        if dois:
            doi = dois[0]
            while doi.endswith((")", ".")):  # Check if the string ends with ")" or "."
                doi = doi[:-1]  # Remove the last character
            return doi

    return np.nan


def search_by_title(title):
    title_encoded = urllib.parse.quote_plus(title)
    term = f'"{title_encoded}"[Title:~1]'
    url = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&"
        f"term={term}&retmode=json"
    )
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"PubMed API returned status code {response.status_code} for {url}")

    data = response.json()
    id_list = data.get("esearchresult", {}).get("idlist", [])

    return id_list[0] if id_list else None


def _add_pmid_pmcid(data_df):
    # Get PMIDs and PMCIDs
    # Drop collections without PMIDs. It either means the DOI is invalid or the paper is not
    # indexed in PubMed.
    data_df["pmid"] = data_df.doi.apply(get_pmid_from_doi)
    data_df = data_df[data_df.pmid.notnull()]
    data_df["pmcid"] = data_df.pmid.apply(get_pmcid_from_pmid)

    return data_df


def _get_col_doi(data_df):
    collections_with_dois = data_df[data_df["DOI"].notnull()][["id", "name", "DOI"]]
    collections_with_dois = collections_with_dois.rename(
        columns={"id": "collection_id", "name": "collection_name", "DOI": "doi"}
    )

    collections_with_dois = _add_pmid_pmcid(collections_with_dois)
    collections_with_dois["source"] = "neurovault"

    return collections_with_dois


def _get_col_doi_meta(data_df):
    # Find DOI in collection description
    data_df["DOI"] = data_df.apply(_look_up_doi, axis=1)
    data_df = data_df.dropna(subset="DOI")

    data_df = data_df[["id", "name", "DOI"]]
    data_df = data_df.rename(
        columns={"id": "collection_id", "name": "collection_name", "DOI": "doi"}
    )

    data_df = _add_pmid_pmcid(data_df)
    data_df["source"] = "metadata"

    return data_df


def _get_col_pmid_title(data_df):
    # Drop collections with names that are too short
    data_df = data_df[data_df.name.notnull()]
    data_df = data_df[data_df["name"].str.len() > 40]

    # Get PMIDs and PMCIDs from title
    data_df["pmid"] = data_df.name.apply(search_by_title)

    data_df = data_df[data_df.pmid.notnull()][["id", "name", "pmid"]]
    data_df = data_df.rename(columns={"id": "collection_id", "name": "collection_name"})

    data_df["pmcid"] = data_df.pmid.apply(get_pmcid_from_pmid)
    data_df["source"] = "pubmed"

    return data_df


def _get_col_pubget(collections_df, data_df, pubget_nv_df, pubget_metadata_df):
    # Convert private_token to collection_id
    collection_ids = pubget_nv_df["collection_id"].to_list()
    pubget_nv_df["collection_id"] = [
        _convert_collection_id(id_, collections_df) for id_ in collection_ids
    ]

    # Get PMIDs and PMCIDs from metadata
    pubget_nv_df = pd.merge(pubget_nv_df, pubget_metadata_df[["pmcid", "pmid", "doi"]], on="pmcid")
    pubget_nv_df = pubget_nv_df.reindex(columns=["pmid", "pmcid", "doi", "collection_id"])
    pubget_nv_df = pubget_nv_df.rename(columns={"doi": "secondary_doi"})
    pubget_nv_df["pmid"] = pubget_nv_df["pmid"].astype("Int64")

    # Some private collections couldnt be mapped to public ones
    pubget_nv_df = pubget_nv_df.dropna(subset=["collection_id"])

    # Get collections found by pubget
    nv_coll = data_df["collection_id"].to_list()
    pubget_nv_coll = pubget_nv_df["collection_id"].to_list()
    matching_ids = np.intersect1d(nv_coll, pubget_nv_coll)

    pubget_mask = ~pubget_nv_df["collection_id"].isin(matching_ids)
    pubget_nv_df = pubget_nv_df[pubget_mask]

    # Select unique collections
    pubget_nv_df = pubget_nv_df.sort_values("pmid")
    pubget_nv_df = pubget_nv_df.drop_duplicates("collection_id", keep="first")

    # Get collection names
    pubget_nv_df = pd.merge(
        pubget_nv_df, collections_df[["id", "name"]], left_on="collection_id", right_on="id"
    )
    pubget_nv_df = pubget_nv_df.rename(columns={"name": "collection_name"})
    pubget_nv_df = pubget_nv_df.drop(columns="id")
    pubget_nv_df["source"] = "pubget"

    return pubget_nv_df


def main(project_dir, neurovault_version, pg_query_id):
    data_dir = op.join(project_dir, "data")
    nv_data_dir = op.join(data_dir, "neurovault", neurovault_version)
    pubget_dir = op.join(data_dir, "pubget_data")
    pubget_query = op.join(pubget_dir, f"query_{pg_query_id}")

    # Load NV data
    collections_df = pd.read_csv(op.join(nv_data_dir, "statmaps_collection.csv"))
    print(f"Found {collections_df.shape[0]} collections")

    # Load pubget data
    pubget_metadata_fn = op.join(pubget_query, "subset_allArticles_extractedData", "metadata.csv")
    pubget_nv_fn = op.join(
        pubget_query,
        "subset_allArticles_extractedData",
        "neurovault_collections.csv",
    )
    pubget_nv_df = pd.read_csv(pubget_nv_fn)
    pubget_metadata_df = pd.read_csv(pubget_metadata_fn)

    # 0. Remove Neuroscout collections
    collections_df = collections_df[collections_df.owner_id != NEUROSCOUT_OWNER_ID]
    print(f"Found {collections_df.shape[0]} collections after removing Neuroscout collections")

    # 1. Get collections with DOIs
    # =================================
    collections_with_dois = _get_col_doi(collections_df)
    print(f"Found {collections_with_dois.shape[0]} collections with DOIs")

    # 2. Find DOI for NeuroVault collections using the metadata
    # ======================================================
    # Get the collections without DOI links
    collections_without_dois = collections_df[
        ~collections_df["id"].isin(collections_with_dois["collection_id"])
    ]
    collections_without_dois = _get_col_doi_meta(collections_without_dois)
    print(f"Found {collections_without_dois.shape[0]} new collections with DOIs from metadata")

    # Concatenate the collections
    collections_with_pmid = pd.concat(
        [collections_with_dois, collections_without_dois], ignore_index=True, sort=False
    )

    # 3. Find PMID for NeuroVault collections using the collection name
    # ======================================================
    collections_missing = collections_df[
        ~collections_df["id"].isin(collections_with_pmid["collection_id"])
    ]
    collections_missing = _get_col_pmid_title(collections_missing)
    print(f"Found {collections_missing.shape[0]} new collections with using the collection name")

    collections_with_pmid = pd.concat(
        [collections_with_pmid, collections_missing], ignore_index=True, sort=False
    )

    # 4. Find NeuroVault collections using pubget search
    # ======================================================
    # Load Pubget data
    pubget_nv_df = _get_col_pubget(
        collections_df,
        collections_with_pmid,
        pubget_nv_df,
        pubget_metadata_df,
    )
    print(f"Found {pubget_nv_df.shape[0]} new collections with using the pubget search")

    # Concatenate the collections
    collections_with_pmid = pd.concat(
        [collections_with_pmid, pubget_nv_df], ignore_index=True, sort=False
    )

    # Add missing collections
    collections_missing = collections_df[
        ~collections_df["id"].isin(collections_with_pmid["collection_id"])
    ][["id", "name"]]
    collections_missing["source"] = "missing"
    collections_missing["pmid"] = np.nan
    collections_missing["pmcid"] = np.nan
    collections_missing["doi"] = np.nan
    collections_missing = collections_missing.rename(
        columns={"id": "collection_id", "name": "collection_name"}
    )

    collections_final_df = pd.concat(
        [collections_with_pmid, collections_missing], ignore_index=True, sort=False
    )

    collections_with_pmid.to_csv(op.join(data_dir, "nv_pmid_collections.csv"), index=False)
    collections_final_df.to_csv(op.join(data_dir, "nv_all_collections.csv"), index=False)

    pmcids = collections_with_pmid["pmcid"].dropna().astype(int).astype(str).unique()
    np.savetxt(op.join(data_dir, "neurovault", "nv-pmcids.txt"), pmcids, fmt="%s")


def _main(argv=None):
    option = _get_parser().parse_args(argv)
    kwargs = vars(option)
    main(**kwargs)


if __name__ == "__main__":
    _main()
