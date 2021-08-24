import json
import os
import click

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


def index_query(filename):
    import shutil
    if os.path.exists(os.environ.get('JINA_WORKSPACE')):
        shutil.rmtree(os.environ.get('JINA_WORKSPACE'))
    f = Flow.load_config('index.yml')
    with f:
        f.post(on='/index', inputs=load_data(filename), request_size=2, show_progress=True)
        f.protocol = 'http'
        f.cors = True
        f.expose_endpoint(
            '/search',
            summary='Search the docs',
            description='example: {"data": [{"text": "信用卡纠纷"}], "parameters": {"top_k": 10}}',
        )
        f.block()

@click.command()
@click.option(
    '--filename',
    '-f',
    type=click.Path(exists=True),
    default='toy-data/case_parse_10.json')
def main(filename):
    config()
    index_query(filename)


if __name__ == "__main__":
    main()
