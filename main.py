import json
import os

from jina import Document, Flow
from executors import IndexSentenceSegmenter, QuerySentenceSegmenter


def config():
    os.environ["JINA_WORKSPACE"] = "./workspace"
    os.environ["JINA_DUMP_PATH_DOC"] = "./workspace/dump_doc"
    os.environ["JINA_DUMP_PATH_CHUNK"] = "./workspace/dump_chunk"
    os.environ["JINA_WORKSPACE_DOC"] = "./workspace/ws_doc"
    os.environ["JINA_WORKSPACE_CHUNK"] = "./workspace/ws_chunk"


def load_data(data_fn='./toy-data/case_parse_100.json'):
    counter = 0
    with open(data_fn, 'r') as f:
        for l in f:
            if not l:
                continue
            doc = Document(json.loads(l))
            try:
                doc.text = doc.tags['_source']['title']
                doc.id = doc.tags['_id']
                counter += 1
                yield doc

            except KeyError as e:
                continue


description='example: {"data": [{"text": "信用卡纠纷申请执行人中国邮政储蓄银行股份有限公司天津武清区支行与被执行人程红玉信用卡纠纷一案"}], "parameters": {"top_k": 10}}'


def index_query(remove_workspace=True):
    import shutil
    if remove_workspace and os.path.exists(os.environ.get('JINA_WORKSPACE')):
        shutil.rmtree(os.environ.get('JINA_WORKSPACE'))
    f = Flow.load_config('index.yml')
    with f:
        f.post(on='/index', inputs=load_data, request_size=2, show_progress=True)
        f.protocol = 'http'
        f.cors = True
        f.block()


def main():
    config()
    index_query()


if __name__ == "__main__":
    main()
