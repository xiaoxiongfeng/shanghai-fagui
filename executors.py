from typing import List, Dict, Iterable, Optional
from itertools import groupby
from gensim.summarization import bm25
import numpy as np

from jina import Document, DocumentArray, Executor, requests
from jina.types.score import NamedScore
import re

import pkuseg

MAX_MODALITIES = 100

from zhon.hanzi import punctuation as chinese_punctuation  # 中文标点符号

chi_punc = '|'.join([c for c in chinese_punctuation])


# 方法2：在切分后，过滤掉split返回的list中的空字符串
# filter_data()函数的功能是：对于一个由string组成的list [str1, str2, str3, ......]，过滤掉那些空字符串''、特殊字符串'\n'，并返回过滤后的新list
def not_break(sen):
    return sen != '\n' and sen != '\u3000' and sen != '' and not sen.isspace()


def filter_data(ini_data):
    # ini_data是由句子组成的string
    new_data = list(filter(not_break, [data.strip() for data in ini_data]))
    return new_data


class IndexSentenceSegmenter(Executor):
    @requests(on='/index')
    def segment(self, docs: DocumentArray, **kwargs):
        if not docs:
            return
        for doc in docs:
            if '_source' not in doc.tags:
                continue
            # paras chunk, we need to filter the paras based on types
            paras = doc.tags['_source'].get('paras', None)
            if paras:
                for para in paras:
                    if not para['content']:
                        continue
                    if para['tag'] in ['书记员', '审判人员', 'judges', 'basics', 'party_info', '附', 'defendant_plea',
                                       'plaintiff_claims', '审判法官', 'court_consider', '当事人信息', '基础信息', '本院认为', '裁判结果',
                                       '调解结果']:
                        continue
                    for p_idx, p in enumerate(para['content'].split('\n')):
                        s_list = filter_data(re.split(r'' + ('[。]'), p))
                        for s_idx, s in enumerate(s_list):
                            s = s.strip()
                            _chunk = Document(
                                text=s[-64:], parent_id=doc.id, modality='paras', location=[p_idx, s_idx])
                            _chunk.tags['para_tag'] = para['tag']
                            doc.chunks.append(_chunk)

            # title chunk
            title = doc.tags['_source'].get('title', None)
            if title:
                title_list = title.split('\n')
                title_text = ''.join(title_list)
                _chunk = Document(
                    text=title_text[-64:],
                    parent_id=doc.id,
                    modality='title',
                )
                for s_idx, s in enumerate(title_list):
                    s_list = filter_data(re.split(r'' + ('[' + chi_punc + ' ' + ']'), s))
                    if len(s_list) <= 1:
                        continue
                    for s_idx, s in enumerate(s_list):
                        s = s.strip()
                        if not s:
                            continue
                        if s == _chunk.text:
                            continue
                        _chunk_chunk = Document(
                            text=s[-64:],
                            parent_id=doc.id,
                            location=[s_idx, s_idx],
                            modality='title_subsentence',
                        )
                        _chunk.chunks.append(_chunk_chunk)
                doc.chunks.append(_chunk)

            # causes chunk
            causes = doc.tags['_source'].get('causes', None)
            if causes:
                for cause in causes:
                    if cause in ['其他', ]:
                        continue
                    if not cause:
                        continue
                    _chunk = Document(
                        text=cause, parent_id=doc.id, modality='causes'
                    )
                    doc.chunks.append(_chunk)

            # docType chunk
            doc_type = doc.tags['_source'].get('docType', None)
            if doc_type:
                _chunk = Document(
                    text=doc_type, parent_id=doc.id, modality='docType'
                )
                doc.chunks.append(_chunk)

            # topCause chunk
            top_cause = doc.tags['_source'].get('topCause', None)
            if top_cause:
                _chunk = Document(
                    text=top_cause, parent_id=doc.id, modality='topCause'
                )
                doc.chunks.append(_chunk)

            # topCause+docType chunk
            if top_cause and doc_type:
                _chunk = Document(
                    text=top_cause + doc_type, parent_id=doc.id, modality='topCause_docType'
                )
                doc.chunks.append(_chunk)

            # trialRound chunk
            trial_round = doc.tags['_source'].get('trialRound', None)
            if trial_round in ['其他', ]:
                trial_round = None
            if trial_round:
                _chunk = Document(
                    text=trial_round, parent_id=doc.id, modality='trialRound'
                )
                doc.chunks.append(_chunk)

            # trialRound+docType
            if trial_round and doc_type:
                _chunk = Document(
                    text=trial_round + doc_type,
                    parent_id=doc.id,
                    modality='trialRound_docType'
                )
                doc.chunks.append(_chunk)

            # topCause+docType chunk
            if trial_round and top_cause and doc_type:
                _chunk = Document(
                    text=trial_round + top_cause + doc_type,
                    parent_id=doc.id,
                    modality='trialRound_topCause_docType'
                )
                doc.chunks.append(_chunk)

            # court chunk
            if doc.tags['_source']['court']:
                court_ = doc.tags['_source']['court']
                _chunk = Document(
                    text=court_, parent_id=doc.id, modality='court'
                )
                doc.chunks.append(_chunk)

        return DocumentArray([d for d in docs if d.chunks])


class QuerySentenceSegmenter(Executor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seg = pkuseg.pkuseg(model_name='news', postag=False)

    @requests(on='/search')
    def segment(self, docs: DocumentArray, **kwargs):
        for doc in docs:
            for s_idx, s in enumerate(doc.text.split(' ')):
                s = s.strip()
                if not s:
                    continue
                _chunk = Document(
                    text=s[:64], parent_id=doc.id, location=[s_idx]
                )
                doc.chunks.append(_chunk)
        return DocumentArray([d for d in docs if d.chunks])


class AggregateRanker(Executor):
    def __init__(
            self,
            default_top_k: int = 5,
            default_traversal_paths: List[str] = ['r'],
            metric: str = 'cosine',
            is_distance: bool = False,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.default_top_k = default_top_k
        self.default_traversal_paths = default_traversal_paths
        self.metric = metric
        if is_distance:
            self.distance_mult = 1
        else:
            self.distance_mult = -1

    @requests(on='/search')
    def rank(self, docs: DocumentArray, parameters: Dict = None, **kwargs):
        top_k = int(parameters.get('top_k', self.default_top_k))

        traversal_paths = parameters.get(
            'traversal_paths', self.default_traversal_paths)

        for doc in docs.traverse_flat(traversal_paths):
            matches_of_chunks = []
            for chunk in doc.chunks:
                matches_of_chunks.extend(chunk.matches)
            groups = groupby(
                sorted(matches_of_chunks, key=lambda d: d.parent_id),
                lambda d: d.parent_id)

            for key, group in groups:
                chunk_match_list = list(group)
                num_modalities = frozenset([m.modality for m in chunk_match_list])
                operands = []
                for m in chunk_match_list:
                    o = NamedScore(
                        op_name=f'{m.location[0]}' if m.location else '',
                        value=m.scores[self.metric].value,
                        ref_id=m.parent_id,
                        description=f'#{m.modality}#{m.text}',
                    )
                    operands.append(o)

                # sort by # of match sources
                match = chunk_match_list[0]
                match.id = chunk_match_list[0].parent_id
                match.scores[self.metric].set_attrs(operands=operands)
                match.scores['num_modalities'] = len(num_modalities) if match.scores[
                                                                            self.metric].value > 1e-5 else MAX_MODALITIES
                match.pop('embedding')
                doc.matches.append(match)

            doc.matches.sort(
                key=lambda d: (
                    -d.scores['num_modalities'].value,
                    self.distance_mult * d.scores[self.metric].value))
            doc.matches = doc.matches[:top_k]

            # trim `chunks` and `tags`
            doc.pop('chunks', 'tags')


class RemoveTags(Executor):
    @requests
    def remove(self, docs, **kwargs):
        for d in docs:
            # continue
            d.pop('tags')


class DebugExecutor(Executor):
    def __init__(self, metric: str = 'l2', *args, **kwargs):
        super().__init__(**kwargs)
        from jina.logging.logger import JinaLogger

        self.logger = JinaLogger(self.__class__.__name__)
        self.metric = metric

    @requests(on='/search')
    def debug(self, docs, **kwargs):
        for i, d in enumerate(docs):
            print(f'[{i}] chunks: {len(d.chunks)}')
            for j, c in enumerate(d.chunks):
                if j >= 3:
                    print(f'\t ...')
                    # assert c.embedding.shape == (768,)
                else:
                    print(f'\t emb shape: {c.embedding.shape}')
                # print(
                #     f'modality: {c.matches[0].parent_id} - {c.matches[0].modality} - {c.matches[0].scores[self.metric].value}'
                # )
                # print(
                #     f'[{i} - {j}]: {len(c.matches)} - [{" ".join([str(m.scores[self.metric].value) for m in c.matches])}]'
                # )


class IndexNameSegmenter(Executor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seg = pkuseg.pkuseg(model_name='news')
        self._punct_pat = re.compile(r'' + ('[' + chi_punc + ' -' + ']'))
        self._name_tags = frozenset(['书记员', '审判人员', 'judges', '审判法官', '当事人信息'])
        self._party_blacklist = frozenset(['二〇', '二O', '二零'])

    @requests(on='/search')
    def segement(self, docs: Optional[DocumentArray], **kwargs):
        if not docs:
            return
        for doc in docs:
            for c in doc.chunks:
                if not c.text:
                    continue
                c.text = ' '.join(self.seg.cut(c.text))

    @requests(on='/index')
    def extract_name(self, docs: Optional[DocumentArray], **kwargs):
        if not docs:
            return
        for doc in docs:
            _source = doc.tags.get('_source', None)
            if not _source:
                continue
            result = []
            # extract party info
            _party = _source.get('party', None)
            result += self._extract_party(_party)
            # extract paras info
            _paras = _source.get('paras', None)
            result += self._extract_paras(_paras)
            doc.text = ' '.join(result)
            doc.modality = 'name'
            doc.pop('tags', 'chunks')

    def _extract_party(self, data_dict: Optional[Dict]):
        result = []
        if not data_dict:
            return result
        for p in data_dict:
            _name = p.get('name', None)
            _type = p.get('type', None)
            if _name:
                result += self.seg.cut(_name)
        return result

    def _extract_paras(self, data_dict: Optional[Dict]):
        result = []
        if not data_dict:
            return result
        for para in data_dict:
            content = para.get('content', None)
            if not content:
                continue
            tag = para.get('tag', None)
            if tag in self._name_tags:
                for c in content.split('\n'):
                    c = c.strip()
                    if not c or len(c) > 10:
                        continue
                    if c[:2] in self._party_blacklist:
                        continue
                    for s in filter_data(self._punct_pat.split(c)):
                        result += self.seg.cut(s)
        return result


class BM25Indexer(Executor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._flush = False
        self._model = None
        self._corpus = []
        self._doc_list = []

    @requests(on='/index')
    def index(self, docs: Optional[DocumentArray], **kwargs):
        if not docs:
            return
        for doc in docs:
            if not doc.text:
                continue
            self._corpus.append(doc.text.split(' '))
            self._doc_list.append(doc)
            self._flush = True

    @requests(on='/search')
    def search(self, docs: Optional[DocumentArray], parameters: dict, **kwargs):
        if not docs:
            return
        if self._model is None or self._flush:
            self._model = bm25.BM25(self._corpus)
            self._flush = False
        top_k = min(len(self._corpus), int(parameters.get('top_k', 10)))
        for doc in docs:
            tokens = []
            for c in doc.chunks:
                tokens += c.text.split(' ')
            scores = np.array(self._model.get_scores(tokens))
            # return top_k from the scores and the Documents
            ind = np.argpartition(scores, -top_k)[-top_k:]
            ind = sorted(ind, key=lambda x: scores[x], reverse=True)
            for idx in ind:
                score = scores[idx]
                if score <= 0:
                    break
                m = self._doc_list[idx]
                m.scores['bm25'] = NamedScore(value=score)
                doc.matches.append(m)


class CaseNumSegmenter(Executor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seg = pkuseg.pkuseg(model_name='news')
        self._char_blacklist = frozenset(chinese_punctuation)

    @requests(on='/index')
    def extract_casenum(self, docs: Optional[DocumentArray], **kwargs):
        if not docs:
            return
        for doc in docs:
            _source = doc.tags.get('_source', None)
            if not _source:
                continue
            _case_num = _source.get('caseNumber', None)
            if not _case_num:
                continue
            doc.text = self._cut(_case_num)
            doc.modality = 'caseNumber'
            doc.pop('tags', 'chunks')

    @requests(on='/search')
    def segment(self, docs: Optional[DocumentArray], **kwargs):
        if not docs:
            return
        for doc in docs:
            for c in doc.chunks:
                if not c.text:
                    continue
                c.text = self._cut(c.text)

    def _cut(self, case_num_str):
        return ' '.join([
            t for t in self.seg.cut(case_num_str)
            if t not in self._char_blacklist])


class ChunkMatchesMerger(Executor):
    @requests(on='/search')
    def merge(self, docs: DocumentArray, parameters: Optional[Dict] = None, **kwargs):
        top_k = int(parameters.get('top_k', 10))
        for doc in docs:
            for chunk in doc.chunks:
                pass


class ChunkFilter(Executor):
    def __init__(self, traversal_paths: Iterable[str] = ('r', ), *args, **kwargs):
        super(ChunkFilter, self).__init__(*args, **kwargs)
        self.traversal_paths = traversal_paths

    @requests(on='/index')
    def filter(self, docs: Optional[DocumentArray] = None, **kwargs):
        if not docs:
            return
        filtered_docs = DocumentArray()
        for d in docs.traverse_flat(self.traversal_paths):
            d.pop('chunks')
            filtered_docs.append(d)
        return filtered_docs
