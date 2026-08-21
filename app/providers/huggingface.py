import json

from huggingface_hub import (
    HfApi,
    hf_hub_download,
    hf_hub_url,
)


class HuggingFaceProvider:

    def __init__(self, token: str | None = None):
        self.token = token

        if token:
            self.api = HfApi(token=token)
        else:
            self.api = HfApi()


    def get_model(
        self,
        repo_id: str,
        revision: str | None = None,
    ):
        return self.api.model_info(
            repo_id=repo_id,
            revision=revision,
            files_metadata=True,
        )


    def get_file_url(
        self,
        repo_id: str,
        filename: str,
        revision: str | None = None,
    ):
        return hf_hub_url(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
        )


    def get_model_index(
        self,
        repo_id: str,
        revision: str | None = None,
    ) -> dict | None:

        try:
            path = hf_hub_download(
                repo_id=repo_id,
                filename="model_index.json",
                revision=revision,
                token=self.token,
            )

        except Exception:
            return None

        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)


    def get_file_metadata(
        self,
        repo_id: str,
        filename: str,
        revision: str | None = None,
    ):

        model = self.get_model(
            repo_id=repo_id,
            revision=revision,
        )

        for sibling in model.siblings:

            if sibling.rfilename == filename:
                return sibling

        raise FileNotFoundError(
            f"File not found in repository: "
            f"{repo_id}/{filename}"
        )
