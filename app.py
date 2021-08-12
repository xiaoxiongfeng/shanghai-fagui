import json
import os
import click

from jina import Document, Flow
from executors import (
    IndexSentenceSegmenter,
    QuerySentenceSegmenter,
    RemoveTags,
    DebugExecutor,
)


def config():
    os.environ['JINA_DUMP_PATH_DOC'] = './workspace/dump_doc'
    os.environ['JINA_DUMP_PATH_CHUNK'] = './workspace/dump_chunk'
    os.environ['JINA_WORKSPACE_DOC'] = './workspace/ws_doc'
    os.environ['JINA_WORKSPACE_CHUNK'] = './workspace/ws_chunk'


def load_data(data_fn='toy-data/case_parse_10.json'):
    counter = 0
    with open(data_fn, 'r') as f:
        for l in f:
            doc = Document(json.loads(l))
            try:
                doc.text = doc.tags['_source']['title']
                doc.id = doc.tags['_id']
                counter += 1
                yield doc
            except KeyError as e:
                continue


def index():
    f_index = Flow.load_config('flows/index.yml')

    with f_index:
        f_index.post(on='/index', inputs=load_data, request_size=16)
        f_index.post(
            on='/dump',
            target_peapod='chunk_indexer',
            parameters={
                'dump_path': os.environ.get('JINA_DUMP_PATH_CHUNK'),
                'shards': 1,
            },
        )
        f_index.post(
            on='/dump',
            target_peapod='doc_indexer',
            parameters={'dump_path': os.environ.get('JINA_DUMP_PATH_DOC'), 'shards': 1},
        )


def query():
    f_query = Flow.load_config('flows/query.yml')
    with f_query:
        results = f_query.post(
            on='/search', inputs=load_data, parameters={'top_k': 3}, return_results=True
        )
    for doc in results[0].docs:
        print(f'query: {doc.id}, {doc.text}')
        for m in doc.matches:
            print(f'+- {m.id}, {m.text}')


def query_restful():
    f_query = Flow.load_config(
        'flows/query.yml',
        override_with={
            'protocol': 'http',
            'cors': True,
            'title': '擎盾科技 Demo',
            'description': 'This is a demo at 擎盾科技',
        },
    )
    f_query.expose_endpoint('/search', summary='Search the docs')
    with f_query:
        f_query.block()


@click.command()
@click.option('--task', '-t', type=click.Choice(['index', 'query', 'query_restful']))
def main(task):
    config()
    if task == 'index':
        index()
    elif task == 'query':
        query()
    elif task == 'query_restful':
        query_restful()


if __name__ == '__main__':
    main()
