from __future__ import unicode_literals

from os import path

import sys
import pyximport
pyximport.install()
sys.path.append(".")
from sense2vec.vectors import VectorMap
from sense2vec.about import __version__
from thinc.neural.util import get_array_module
from math import sqrt


def load(vectors_path, dim):
    if not path.exists(vectors_path):
        raise IOError("Can't find data directory: {}".format(vectors_path))
    vector_map = VectorMap(dim)
    vector_map.load(vectors_path)
    return vector_map


def transform_doc(doc):
    """
    Transform a spaCy Doc to match the sense2vec format: merge entities
    into one token and merge noun chunks without determiners.
    """
    #if not doc.is_tagged:
    #    raise ValueError("Can't run sense2vec: document not tagged.")
    for ent in doc.ents:
        ent.merge(tag=ent.root.tag_, lemma=ent.root.lemma_,
                  ent_type=ent.label_)
    for np in doc.noun_chunks:
        while len(np) > 1 and np[0].dep_ not in ('advmod', 'amod', 'compound'):
            np = np[1:]
        np.merge(tag=np.root.tag_, lemma=np.root.lemma_,
                ent_type=np.root.ent_type_)
    return doc

class Sense2VecComponent(object):
    """
    spaCy v2.0 pipeline component.

    USAGE:
        >>> import spacy
        >>> from sense2vec import Sense2VecComponent
        >>> nlp = spacy.load('en')
        >>> s2v = Sense2VecComponent('/path/to/model')
        >>> nlp.add_pipe(s2v)
        >>> doc = nlp(u"A text about natural language processing.")
        >>> assert doc[3].text == 'natural language processing'
        >>> assert doc[3]._.in_s2v
        >>> print(doc[3]._.s2v_most_similar(20))
    """
    name = 'sense2vec'

    def __init__(self, vectors_path, dim=300):
        self.s2v = load(vectors_path, dim)
        self.first_run = True

    def __call__(self, doc):
        if self.first_run:
            self.init_component(doc)
            self.first_run = False
        doc = transform_doc(doc)
        return doc

    def init_component(self, doc):
        # initialise the attributes here only if the component is added to the
        # pipeline and used – otherwise, tokens will still get the attributes
        # even if the component is only created and not added
        Doc = doc.__class__
        Token = doc[0].__class__
        Span = doc[:1].__class__
        Doc.set_extension('s2v_similarity', method=lambda t, n: self.s2v_doc_similarity(t, n), force=True)
        Token.set_extension('in_s2v', getter=lambda t: self.in_s2v(t), force=True)
        Token.set_extension('s2v_freq', getter=lambda t: self.s2v_freq(t), force=True)
        Token.set_extension('s2v_vec', getter=lambda t: self.s2v_vec(t), force=True)
        Token.set_extension('s2v_most_similar', method=lambda t, n: self.s2v_most_sim(t, n), force=True)
        Token.set_extension('s2v_similarity', method=lambda t, n: self.s2v_similarity(t, n), force=True)
        Span.set_extension('in_s2v', getter=lambda s: self.in_s2v(s), force=True)
        Span.set_extension('s2v_freq', getter=lambda s: self.s2v_freq(s), force=True)
        Span.set_extension('s2v_vec', getter=lambda s: self.s2v_vec(s), force=True)
        Span.set_extension('s2v_most_similar', method=lambda s, n: self.s2v_most_sim(s, n), force=True)
        Span.set_extension('s2v_similarity', method=lambda s, n: self.s2v_similarity(s, n), force=True)

    def in_s2v(self, obj):
        return self._get_query(obj) in self.s2v

    def s2v_freq(self, obj):
        freq, _ = self.s2v[self._get_query(obj)]
        return freq

    def s2v_vec(self, obj):
        _, vector = self.s2v[self._get_query(obj)]
        return vector
        
    def s2v_most_sim(self, obj, n_similar=10):
        _, vector = self.s2v[self._get_query(obj)]
        words, scores = self.s2v.most_similar(vector, n_similar)
        words = [word.replace('_', ' ') for word in words]
        words = [tuple(word.rsplit('|', 1)) for word in words]
        return list(zip(words, scores))

    def s2v_similarity(self, obj1, obj2):
        _, vector1 = self.s2v[self._get_query(obj1)]
        _, vector2 = self.s2v[self._get_query(obj2)]
        return self.s2v.similarity(vector1, vector2)

    def _get_query(self, obj):
        # no pos_ and label_ shouldn't happen – unless it's an unmerged
        # non-entity Span (in which case we just use the root's pos)
        pos = obj.pos_ if hasattr(obj, 'pos_') else obj.root.pos_
        sense = obj.ent_type_ if (obj.ent_type_) else pos
        return obj.text.replace(' ', '_') + '|' + sense
        
    def s2v_doc_similarity(self, obj1, other):
        """Make a semantic similarity estimate. The default estimate is cosine
        similarity using an average of word vectors.
        other (object): The object to compare with. By default, accepts `Doc`,
            `Span`, `Token` and `Lexeme` objects.
        RETURNS (float): A scalar similarity score. Higher is more similar.
        DOCS: https://spacy.io/api/doc#similarity
        """
        vector1 = self.get_s2v_doc_vector(obj1)
        vector2 = self.get_s2v_doc_vector(other)
        xp = get_array_module(vector1)
        return xp.dot(vector1, vector2) / (self.vector_norm(vector1) * self.vector_norm(vector2))

    def get_s2v_doc_vector(self, doc):
        return sum([self.s2v_vec(t) for t in doc if self.in_s2v(t)]) / len(doc)

    def vector_norm(self, vector):
        """The L2 norm of the document's vector representation.
        RETURNS (float): The L2 norm of the vector representation.
        DOCS: https://spacy.io/api/doc#vector_norm
        """
        norm = 0.0
        for value in vector:
            norm += value * value
        return sqrt(norm) if norm != 0 else 0