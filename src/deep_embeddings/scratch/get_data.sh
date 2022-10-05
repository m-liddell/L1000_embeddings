#URLS from https://clue.io/releases/data-dashboard

#GSE92742 level 4
wget "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742_Broad_LINCS_Level4_ZSPCINF_mlr12k_n1319138x12328.gctx.gz"
gunzip GSE92742_Broad_LINCS_Level4_ZSPCINF_mlr12k_n1319138x12328.gctx.gz

wget "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742_Broad_LINCS_gene_info.txt.gz"
gunzip GSE92742_Broad_LINCS_gene_info.txt.gz

wget "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742_Broad_LINCS_inst_info.txt.gz"
gunzip GSE92742_Broad_LINCS_inst_info.txt.gz

wget -r "ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742_Broad_LINCS_cell_info.txt.gz"
gunzip GSE92742_Broad_LINCS_cell_info.txt.gz

#GSE70138 level 4
wget "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE70nnn/GSE70138/suppl/GSE70138_Broad_LINCS_Level4_ZSPCINF_mlr12k_n345976x12328_2017-03-06.gctx.gz"
gunzip GSE70138_Broad_LINCS_Level4_ZSPCINF_mlr12k_n345976x12328_2017-03-06.gctx.gz

wget "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE70nnn/GSE70138/suppl/GSE70138_Broad_LINCS_gene_info_2017-03-06.txt.gz"
gunzip GSE70138_Broad_LINCS_gene_info_2017-03-06.txt.gz

wget "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE70nnn/GSE70138/suppl/GSE70138_Broad_LINCS_inst_info_2017-03-06.txt.gz"
gunzip GSE70138_Broad_LINCS_inst_info_2017-03-06.txt.gz

wget "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE70nnn/GSE70138/suppl/GSE70138_Broad_LINCS_cell_info_2017-04-28.txt.gz"
gunzip GSE70138_Broad_LINCS_cell_info_2017-04-28.txt.gz

LINC 2020 beta
wget "https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/level4/level4_beta_trt_cp_n1805898x12328.gctx"
wget "https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/geneinfo_beta.txt"
wget "https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/instinfo_beta.txt"
