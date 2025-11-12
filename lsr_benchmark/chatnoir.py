from pathlib import Path
from ir_datasets.formats import BaseDocs
from ir_datasets import Dataset
from ir_datasets.formats import TrecQrels, TrecQueries
from ir_datasets.util import StringFile
from ir_datasets import registry

def register_subsample_from_chatnoir(chatnoir_index: str, qrels_file: Path, topics_file: Path, ir_datasets_id: str):
    if ir_datasets_id in registry:
        return

    from chatnoir_api.irds import ChatNoirDocsStore
    class ChatNoirDocs(BaseDocs):
        def docs_store(self):
            return ChatNoirDocsStore(chatnoir_index)
    docs = ChatNoirDocs()
    qrels = TrecQrels(StringFile(open(qrels_file, "r").read()), {})
    topics = TrecQueries(StringFile(open(topics_file, "r").read()))
    dataset = Dataset(docs, qrels, topics)
    registry.register(ir_datasets_id, dataset)
    

