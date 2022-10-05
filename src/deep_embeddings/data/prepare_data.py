#from __future__ import annotations
from pathlib import Path
from typing import NamedTuple, List, Dict
import logging
import pandas as pd
from cmapPy.pandasGEXpress import parse, concat, GCToo

class CmapDatasetPath(NamedTuple):
    gene_info: Path
    inst_info: Path
    gene_expression: Path

class CmapDataset(NamedTuple):
    gene_info: pd.DataFrame
    inst_info: pd.DataFrame
    gene_expression: GCToo

def read_treated_and_landmark(cmap_dataset_paths: CmapDatasetPath) -> CmapDataset:
    """
    Reads in only landmark genes from only treated samples from level 4 GCTX data
    Reads in corresponding metadata and adds this to the GCToo object
    """
    gene_info = pd.read_csv(cmap_dataset_paths.gene_info, sep="\t", dtype=str)
    landmark_gene_info = gene_info[gene_info["pr_is_lm"] == "1"]

    inst_info = pd.read_csv(cmap_dataset_paths.inst_info, sep="\t", dtype=str)
    treated_inst = inst_info[(inst_info['pert_type'] == "trt_cp") & ~(inst_info['pert_dose_unit'] == "-666")]

    gene_expression = parse.parse(str(cmap_dataset_paths.gene_expression), 
        rid = landmark_gene_info["pr_gene_id"],
        cid = treated_inst["inst_id"])

    #add metadata
    gene_expression.col_metadata_df = treated_inst.copy().set_index("inst_id")

    logging.info(f"Dimensions of data read from '{cmap_dataset_paths.gene_expression}' are {gene_expression.data_df.shape}")
    assert gene_expression.data_df.shape[0] == 978

    return CmapDataset(gene_info, treated_inst, gene_expression)

def gene_symbol_to_id(gene_info: pd.DataFrame) -> Dict:
    """
    Returns a dictionary mapping gene ids to gene symbols
    """
    return dict(zip(gene_info["pr_gene_id"], gene_info["pr_gene_symbol"]))

def get_gene_symbol_to_id(gene_info: List[pd.DataFrame]) -> Dict:
    """
    Checks all the gene ids and gene symbols for all datasets match
    Returns dictionary mapping them if true
    """
    gene_symbol_to_id_lookup = [gene_symbol_to_id(d) for d in gene_info]
    assert all(l == gene_symbol_to_id_lookup[0] for l in gene_symbol_to_id_lookup) == True
    return gene_symbol_to_id_lookup[0]

def make_merged_df(datasets: list, gene_symbol_to_id_lookup) -> pd.DataFrame:
    """
    Concants expression dataframes and renames gene ids ready for model training
    """
    datasets_concat = concat.hstack(datasets)
    logging.info(f"Dimensions of data after concatinating datasets are {datasets_concat.data_df.shape}")

    df = datasets_concat.data_df.transpose(copy=True)   
    df = df.rename(columns=gene_symbol_to_id_lookup)
    df = df.join(datasets_concat.col_metadata_df)
    return df

def clean_data(GSE92742_paths, GSE70138_paths):
    GSE92742 = read_treated_and_landmark(GSE92742_paths)
    GSE70138 = read_treated_and_landmark(GSE70138_paths)

    gene_symbol_to_id_lookup = get_gene_symbol_to_id([GSE92742.gene_info, GSE70138.gene_info])
    return make_merged_df([GSE92742.gene_expression, GSE70138.gene_expression], gene_symbol_to_id_lookup)