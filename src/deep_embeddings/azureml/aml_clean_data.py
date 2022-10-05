from pathlib import Path
from azureml.core import Run
from deep_embeddings.data.prepare_data import clean_data, CmapDatasetPath

def main() -> None:
    run = Run.get_context()
    data_path = Path("../../../data")
    raw_data_path = data_path / "raw"
    clean_data_path = data_path / "clean"

    GSE92742_paths = CmapDatasetPath(
        raw_data_path / "GSE92742_Broad_LINCS_gene_info.txt",
        raw_data_path / "GSE92742_Broad_LINCS_inst_info.txt",
        raw_data_path / "GSE92742_Broad_LINCS_Level4_ZSPCINF_mlr12k_n1319138x12328.gctx"
    )
    GSE70138_paths = CmapDatasetPath(
        raw_data_path / "GSE70138_Broad_LINCS_gene_info_2017-03-06.txt",
        raw_data_path / "GSE70138_Broad_LINCS_inst_info_2017-03-06.txt",
        raw_data_path / "GSE70138_Broad_LINCS_Level4_ZSPCINF_mlr12k_n345976x12328_2017-03-06.gctx"
    )
    #TODO add this dataset after paper results are reproduced
    #lincs_2020_beta_paths = CmapDatasetPath(
    #    raw_data_path / "geneinfo_beta.txt",
    #    raw_data_path / "instinfo_beta.txt",
    #    raw_data_path / "level4_beta_trt_cp_n1805898x12328.gctx"
    #)

    df = clean_data(GSE92742_paths, GSE70138_paths)
    mounted_output_dir = Path(run.output_datasets["clean"])
    mounted_output_dir.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(mounted_output_dir / "clean.parquet")

if __name__ == "__main__":
    main()
    