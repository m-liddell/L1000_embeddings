# %%
from cmapPy.pandasGEXpress import parse, concat
import pandas as pd

# %%
gene_info_GSE92742 = pd.read_csv("data/raw/GSE92742_Broad_LINCS_gene_info.txt", sep="\t", dtype=str)
landmark_gene_row_ids_GSE92742 = gene_info_GSE92742["pr_gene_id"][gene_info_GSE92742["pr_is_lm"] == "1"]

# %%
inst_info_GSE92742 = pd.read_csv("data/raw/GSE92742_Broad_LINCS_inst_info.txt", sep="\t", dtype=str)

# %%
treated_inst_GSE92742 = inst_info_GSE92742[inst_info_GSE92742['pert_type'].str.startswith('trt')] #includes genetic pertubagens
#treated_inst_GSE92742 = inst_info_GSE92742[(inst_info_GSE92742['pert_type'] == "trt_cp") & ~(inst_info_GSE92742['pert_dose_unit'] == "-666")]

# %%
gctx_GSE92742 = parse.parse("data/raw/GSE92742_Broad_LINCS_Level4_ZSPCINF_mlr12k_n1319138x12328.gctx", 
    rid = landmark_gene_row_ids_GSE92742,
    cid = treated_inst_GSE92742["inst_id"])

gctx_GSE92742.data_df.shape #check this is 978

# %%
treated_inst_GSE92742 = treated_inst_GSE92742.copy().set_index("inst_id")
gctx_GSE92742.col_metadata_df = treated_inst_GSE92742

# %%
gene_info_GSE70138 = pd.read_csv("data/raw/GSE70138_Broad_LINCS_gene_info_2017-03-06.txt", sep="\t", dtype=str)
landmark_gene_row_ids_GSE70138 = gene_info_GSE70138["pr_gene_id"][gene_info_GSE70138["pr_is_lm"] == "1"]

# %%
inst_info_GSE70138 = pd.read_csv("data/raw/GSE70138_Broad_LINCS_inst_info_2017-03-06.txt", sep="\t", dtype=str)

# %%
treated_inst_GSE70138 = inst_info_GSE70138[inst_info_GSE70138['pert_type'].str.startswith('trt')] #includes genetic pertubagens
#treated_inst_GSE70138 = inst_info_GSE70138[(inst_info_GSE70138['pert_type'] == "trt_cp") & ~(inst_info_GSE70138['pert_dose_unit'] == "-666")]

# %%
gctx_GSE70138 = parse.parse("data/raw/GSE70138_Broad_LINCS_Level4_ZSPCINF_mlr12k_n345976x12328_2017-03-06.gctx", 
    rid = landmark_gene_row_ids_GSE70138,
    cid = treated_inst_GSE70138["inst_id"])

gctx_GSE70138.data_df.shape #check this is 978
# %%
treated_inst_GSE70138 = treated_inst_GSE70138.copy().set_index("inst_id")
gctx_GSE70138.col_metadata_df = treated_inst_GSE70138

# %%
gctx_concat = concat.hstack([gctx_GSE92742, gctx_GSE70138])
gctx_concat.data_df.shape

# %%
df = gctx_concat.data_df.transpose(copy=True)

gene_symbol_to_id = dict(zip(gene_info_GSE92742["pr_gene_id"], gene_info_GSE92742["pr_gene_symbol"]))
df = df.rename(columns=gene_symbol_to_id)
# %%
df = df.join(gctx_concat.col_metadata_df)

# %%