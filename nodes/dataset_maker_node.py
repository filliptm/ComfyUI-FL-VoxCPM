import logging
from comfy_api.latest import io

from ..modules.dataset_utils import create_jsonl_dataset

logger = logging.getLogger(__name__)

VoxCPMDataset = io.Custom("VOXCPM_DATASET")


class FL_VoxCPM_DatasetMaker(io.ComfyNode):
    CATEGORY = "FL/VoxCPM/Training"

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="FL_VoxCPM_DatasetMaker",
            display_name="FL VoxCPM Dataset Maker",
            category=cls.CATEGORY,
            description="Creates a JSONL dataset from a folder of audio files and text transcripts.",
            inputs=[
                io.String.Input("audio_directory", default="", tooltip="Path to directory containing .wav, .mp3, or .flac files with matching .txt files."),
                io.String.Input("output_filename", default="train.jsonl", tooltip="Name of the output JSONL file."),
            ],
            outputs=[
                VoxCPMDataset.Output(display_name="Dataset Path"),
            ],
        )

    @classmethod
    def execute(cls, audio_directory, output_filename):
        try:
            dataset_path = create_jsonl_dataset(audio_directory, output_filename)
            return io.NodeOutput(dataset_path)
        except Exception as e:
            logger.error(f"Dataset creation failed: {e}")
            raise e
