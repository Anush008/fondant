import logging
import re
import sys

import dask
import dask.dataframe as dd
import pandas as pd
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from llama_index import GoogleDocsReader

from fondant.component import DaskLoadComponent
from fondant.component_spec import ComponentSpec

logger = logging.getLogger(__name__)

dask.config.set({"dataframe.convert-string": False})


class LoadFromGoogleDocs(DaskLoadComponent):
    def __init__(
        self,
        spec: ComponentSpec,
        folder_url: list,
        docs_url: list,
        is_folder: bool = True,
    ) -> None:
        self.is_folder = is_folder
        if self.is_folder:
            self.urls = folder_url
        else:
            self.urls = docs_url
        self.spec = spec

    def extract_id_from_url(
        self, urls, pattern_folder=r"folders/([^/]+)", pattern_doc=r"d/([^/]+)/"
    ) -> list:
        if self.is_folder:
            pattern = pattern_folder
        else:
            pattern = pattern_doc

        result = []
        for url in urls:
            match = re.search(pattern, url)
            if match:
                result.append(match.group(1))

        return result

    def load(self) -> dd.DataFrame:
        # 1) define the scope
        SCOPES = [
            "https://www.googleapis.com/auth/drive.metadata.readonly",
            #   "https://www.googleapis.com/auth/documents.readonly"
        ]

        flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
        creds = flow.run_local_server(port=0)
        service = build("drive", "v3", credentials=creds)
        # 2) create document ids
        logger.info("Creating document ids...")
        ids = self.extract_id_from_url(urls=self.urls)
        if self.is_folder:
            # retrieve doc ids from Google API
            document_ids = []
            for folder_id in ids:
                results = (
                    service.files()
                    .list(
                        q=f"'{folder_id}' in parents and mimeType='application/vnd.google-apps.document' and trashed = false",
                        includeItemsFromAllDrives=True,
                        supportsAllDrives=True,
                        fields="nextPageToken, files(id, name)",
                    )
                    .execute()
                )
                items = results.get("files", [])

                for item in items:
                    document_ids.append(item["id"])
        else:
            document_ids = ids

        # 3) Load data
        logger.info("Loading the data from docs files...")
        documents = GoogleDocsReader().load_data(document_ids=document_ids)

        # 4) Create Dask Dataframe
        logger.info("Creating Dataframe...")
        columns_list = []
        for doc in documents:
            columns = {}
            columns["text_chunk_id"] = doc.id_
            columns["text_doc_id"] = doc.metadata["document_id"]
            columns["text_data"] = doc.text
            columns["text_hash"] = doc.hash
            columns_list.append(columns)

        df = pd.DataFrame(columns_list)

        dask_dataframe = dd.from_pandas(df, npartitions=1)

        return dask_dataframe
