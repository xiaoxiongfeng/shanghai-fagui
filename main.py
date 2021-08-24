import json
import os
import click

from jina import Document, Flow
from executors import (
    IndexSentenceSegmenter,
    QuerySentenceSegmenter,
    AggregateRanker,
)

def config():
    os.environ["JINA_WORKSPACE"] = "./workspace"
    os.environ["JINA_DUMP_PATH_DOC"] = "./workspace/dump_doc"
    os.environ["JINA_DUMP_PATH_CHUNK"] = "./workspace/dump_chunk"
    os.environ["JINA_WORKSPACE_DOC"] = "./workspace/ws_doc"
    os.environ["JINA_WORKSPACE_CHUNK"] = "./workspace/ws_chunk"


def load_data(data_fn='./toy-data/case_parse_10.json'):
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


def create_index_flow():
    flow = (
        Flow()
        .add(uses=IndexSentenceSegmenter, name='segmenter')
        .add(
            name='encoder',
            uses='jinahub://TransformerTFTextEncoder',
            parallel=8,
            uses_with={
                'pretrained_model_name_or_path': 'hfl/chinese-legal-electra-small-generator',
                'on_gpu': False,
                'default_batch_size': 128,
                'max_length': 256,
                'default_traversal_paths': ['c', 'cc']})
        # .add(
        #     name='chunk_indexer',
        #     uses='jinahub://PostgreSQLStorage',
        #     uses_with={
        #         'table': 'chunk_indexer_legal_electra_table_court3',
        #         'default_traversal_paths': ['c']},
        #     uses_metas={'workspace': os.environ["JINA_WORKSPACE_CHUNK"]})
        # .add(
        #     name='doc_indexer',
        #     uses='jinahub://PostgreSQLStorage',
        #     uses_with={
        #         'table': 'doc_indexer_table_court3',
        #         'default_traversal_paths': ['r']},
        #     uses_metas={'workspace': os.environ["JINA_WORKSPACE_DOC"]},
        #     needs='gateway')
        # .add(name='joiner', needs=['chunk_indexer', 'doc_indexer'])
    )

    return flow


def create_query_flow():
    flow = (
        Flow()
        .add(name='segmenter', uses=QuerySentenceSegmenter)
        .add(
            name='encoder',
            uses='jinahub://TransformerTFTextEncoder',
            parallel=1,
            uses_with={
                'pretrained_model_name_or_path': 'hfl/chinese-legal-electra-small-generator',
                'on_gpu': False,
                'max_length': 256,
                'default_batch_size': 32,
                'default_traversal_paths': ['c']})
        .add(
            name='chunk_vec_indexer',
            uses='jinahub://FaissSearcher',
            timeout_ready=-1,
            uses_with={
                'index_key': 'HNSW32',
                'requires_training': False,
                'prefech_size': 10000,
                'metric': 'inner_product',
                'normalize': True,  # i.e., cosine metric
                'is_distance': False,
                'dump_path': os.environ["JINA_DUMP_PATH_CHUNK"],
                'default_traversal_paths': ['c']},
            uses_metas={'workspace': os.environ["JINA_WORKSPACE_CHUNK"]})
        .add(
            name='chunk_kv_indexer',
            uses='jinahub://PostgreSQLStorage',
            uses_with={
                'table': 'chunk_indexer_legal_electra_table_court3',
                'default_traversal_paths': ['cm']},
            uses_metas={'workspace': os.environ["JINA_WORKSPACE_CHUNK"]})
        .add(
            name='ranker',
            uses='AggregateRanker',
            uses_with={
                'metric': 'inner_product',
                'is_distance': False,
                'default_traversal_paths': ['r',]})
        .add(
            name='doc_kv_indexer',
            uses='jinahub://PostgreSQLStorage',
            uses_with={
                'table': 'doc_indexer_table_court3',
                'default_traversal_paths': ['m']},
            uses_metas={'workspace': os.environ["JINA_WORKSPACE_DOC"]}
        )
    )
    return flow


def index():
    import shutil

    if os.path.exists(os.environ.get('JINA_WORKSPACE')):
        shutil.rmtree(os.environ.get('JINA_WORKSPACE'))
    f_index = create_index_flow()
    with f_index:
        print(f'==> STEP [1/2]: indexing data ...')
        resp = f_index.post(on='/index', inputs=load_data, request_size=1, show_progress=True, return_results=True)
        for r in resp:
            for doc in r.docs:
                print(f'{doc.id}: {len(doc.chunks)}')
                for c in doc.chunks:
                    print(f'  +- {c.id}: {c.text}, {c.tags}, {c.modality}')
                    if c.chunks:
                        for cc in c.chunks:
                            print(f'    +- {cc.id}: {cc.text}, {cc.tags}, {cc.modality}')

        # print(f'==> STEP [2/2]: dumping chunk data ...')
        # f_index.post(
        #     on='/dump',
        #     target_peapod='chunk_indexer',
        #     parameters={
        #         'dump_path': os.environ.get('JINA_DUMP_PATH_CHUNK'),
        #         'shards': 1,
        #     },
        # )

def query():
    f_query = create_query_flow()
    with f_query:
        results = f_query.post(
            on='/search',
            inputs=Document(
                # text=f'我发生了交通事故,我骑自行车，对方是小轿车，属于工伤，在交通事故中我是负同等责任，现已申请工伤待遇，并接受治疗，应该如何寻找理赔？'
                # text = '海口市美兰区人民检察院与被申请人陈东旭强制医疗一案刑事决定书'
                text = '西藏自治区类乌齐县人民法院'
                # text = '甘肃省西和县人民法院'
                # text = '甘肃省榆中县人民法院'
                # text = '黑龙江省佳木斯市中级人民法院'
                # text = '黑龙江省齐齐哈尔市中级人民法院'
                # text = '海南省海口市秀英区人民法院'
                # text = '关某与王某华一审民事裁定书'
                # text = '一审刑事判决书'
                # text = '天津市静海区人民法院'
                # text = '北京市丰台区人民法院'
                # text = '中国人民财产保险股份有限公司北京市分公司一审民事判决书'
            ),
            parameters={'top_k': 3, 'key_words': '危险驾驶罪'},
            return_results=True,
        )

    for doc in results[0].docs:
        print(f'Query: {doc.id}, {doc.text}')
        for i, m in enumerate(doc.matches):
            print(
                f'+- [{i}] ({m.scores["inner_product"].value}) {m.id}\n\t法院：{m.tags["_source"]["court"]}\n\t标题：{m.tags["_source"]["title"]}'
            )
            input('Enter to continue...')


def update():
    f_index = create_index_flow()
    with f_index:
        print(f'==> STEP [1/2]: update data ...')
        f_index.post(on='/update', inputs=load_data, request_size=1, show_progress=True)

        print(f'==> STEP [2/2]: dumping chunk data ...')
        f_index.post(
            on='/dump',
            target_peapod='chunk_indexer',
            parameters={
                'dump_path': os.environ.get('JINA_DUMP_PATH_CHUNK'),
                'shards': 1,
            },
        )


def delete():
    f_index = create_index_flow()
    with f_index:
        print(f'==> STEP [1/2]: delete data ...')
        f_index.post(on='/delete', inputs=load_data, request_size=1, show_progress=True)

        print(f'==> STEP [2/2]: dumping chunk data ...')
        f_index.post(
            on='/dump',
            target_peapod='chunk_indexer',
            parameters={
                'dump_path': os.environ.get('JINA_DUMP_PATH_CHUNK'),
                'shards': 1,
            },
        )


def query_restful(port_expose='47678'):
    f_query = create_query_flow()
    f_query._update_args(
        None,
        protocol='http',
        port_expose=port_expose,
        cors=True,
        title='擎盾科技 Demo',
        description='This is a demo at 擎盾科技',
        no_debug_endpoints=True,
    )
    f_query.expose_endpoint(
        '/search',
        summary='Search the docs',
        description='example: {"data": [{"text": "信用卡纠纷申请执行人中国邮政储蓄银行股份有限公司天津武清区支行与被执行人程红玉信用卡纠纷一案"}], "parameters": {"top_k": 10}}',
    )
    with f_query:
        f_query.block()


@click.command()
@click.option(
    "--task",
    "-t",
    type=click.Choice(['index', 'query', 'update', 'delete', 'query_restful']),
)
def main(task):
    config()
    index()
    # query()
    # if task == "index":
    #     index()
    # elif task == "query":
    #     query()
    # elif task == 'update':
    #     update()
    # elif task == 'delete':
    #     delete()
    # elif task == 'query_restful':
    #     query_restful()


if __name__ == "__main__":
    main()
