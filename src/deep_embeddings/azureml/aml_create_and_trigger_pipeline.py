from azureml.core import Dataset, Workspace
from azureml.core.datastore import Datastore
from azureml.core.compute import ComputeTarget
from azureml.core.environment import CondaDependencies, Environment
from azureml.core.experiment import Experiment
from azureml.core.runconfig import RunConfiguration
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.data.output_dataset_config import OutputFileDatasetConfig
from azureml.core.runconfig import RunConfiguration
from azureml.pipeline.core.pipeline import Pipeline
from azureml.pipeline.steps import PythonScriptStep
from azureml.exceptions import UserErrorException
from pathlib import Path
import os
from deep_embeddings.azureml import aml_clean_data
import deep_embeddings

def create_and_trigger_pipeline() -> None:

    #TODO abstract out path as they're used elsewhere
    data_path = Path("../../../data")
    raw_data_path = data_path / "raw"
    clean_data_path = data_path / "clean"
    src_dir = Path(deep_embeddings.__file__).parent.parent
    clean_m_path = Path(aml_clean_data.__file__).relative_to(src_dir)

    ws = Workspace.from_config() #assumes this already exists

    #connect datastore
    blob_datastore_name='cmap' # Name of the datastore to workspace
    container_name=os.getenv("BLOB_CONTAINER", "cmap") # Name of Azure blob container
    account_name=os.getenv("BLOB_ACCOUNTNAME", "bnlwestgunileverml00005") # Storage account name
    account_key=os.getenv("BLOB_ACCOUNT_KEY", "") # Storage account access key

    try:
        blob_datastore = Datastore.get(ws, blob_datastore_name)
        print("Found Blob Datastore with name: %s" % blob_datastore_name)
    except UserErrorException:
        blob_datastore = Datastore.register_azure_blob_container(
            workspace=ws,
            datastore_name=blob_datastore_name,
            account_name=account_name,
            container_name=container_name,
            account_key=account_key)
        print("Registered blob datastore with name: %s" % blob_datastore_name)

    #create dataset
    datastore_paths = [(blob_datastore, raw_data_path)]
    try:
        cmap_dataset = Dataset.get_by_name(ws, name='cmap_data')
    except UserErrorException:
        cmap_dataset = Dataset.File.from_files(path=datastore_paths)
        cmap_dataset = cmap_dataset.register(workspace=ws, 
                            name='cmap_data',
                            description='CMap data',
                            tags = {'format':'GCTX'},
                            create_new_version=True)
        print('Dataset registered.')

    #create environment
    try:
        embeddings_env = Environment.get(workspace=ws, name="deep_embeddings")
    except UserErrorException:
        embeddings_env = Environment(name="deep_embeddings")
        conda_dep = CondaDependencies()
        conda_dep.add_channel("bioconda")
        conda_dep.add_conda_package("cmappy")
        embeddings_env.python.conda_dependencies = conda_dep
        embeddings_env.register(workspace=ws)
        
    #define cluster
    cluster_name = "deep-embeddings"
    try:
        pipeline_cluster = ComputeTarget(workspace=ws, name=cluster_name)
        print('Found existing cluster, use it.')
    except ComputeTargetException:
        try:
            compute_config = AmlCompute.provisioning_configuration(vm_size='Standard_D64s_v3', max_nodes=1)
            pipeline_cluster = ComputeTarget.create(ws, cluster_name, compute_config)
            pipeline_cluster.wait_for_completion(show_output=True)
        except Exception as ex:
            print(ex)

    clean_data = (
        OutputFileDatasetConfig(
            name="clean",
            destination=(blob_datastore, clean_data_path),
        )
        .as_upload(overwrite=True)
        .register_on_complete(name="clean")
    )

    run_with_env = RunConfiguration()
    run_with_env.environment = embeddings_env

    #train_m_path = Path(aml_train_model.__file__).relative_to(src_dir)
    clean_step = PythonScriptStep(
        name="clean data",
        script_name=str(clean_m_path),
        source_directory=src_dir,
        inputs=[cmap_dataset.as_named_input("cmap_data").as_mount()],
        outputs=[clean_data],
        compute_target=pipeline_cluster,
        runconfig=run_with_env,
        allow_reuse=True
    )
    #train_step = PythonScriptStep(
    #    name="train_model",
    #    script_name=str(train_m_path),
    #    source_directory=src_dir,
    #    runconfig=RunConfiguration(),
    #    arguments=["--epochs", "5"],
    #    inputs=[clean_data.as_input()],
    #    outputs=[],
    #    compute_target=ComputeTarget(workspace=ws, name="deep_embeddings"),
    #    allow_reuse=True,
    #)

    exp = Experiment(workspace=ws, name="clean_data")
    pipeline = Pipeline(ws, steps=[clean_step])
    run = pipeline.submit(experiment_name=exp.name)
    run.wait_for_completion(raise_on_error=True)

if __name__ == "__main__":
    create_and_trigger_pipeline()
