import gensim, logging
import plac
import os, sys
sys.path.append("..")
from sense2vec.vectors import VectorMap

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

@plac.annotations(
    size=("Embedding size", "positional", None, int),
    window=("Window size", "positional", None, int),
    min_count=("Min frequency count", "positional", None, int),
    workers=("Number of workers", "positional", None, int),
    negative=("Negative sampling count", "positional", None, int),
    epochs=("Number of training epochs", "positional", None, int),
    path=("Input path containing the data", "positional", None, str),
    out_path=("Output path to save the model to", "positional", None, str))
def main(size, window, min_count, workers, negative, epochs, path, out_path):
    ## Use this if you want FastText instead of w2v. Our experience is that FT (with char N-grams)
    ## performs rather bad because it (strongly) emphasizes POS-tags.
    #w2v_model = gensim.models.FastText(
    #    size=size,
    #    window=window,
    #    min_count=min_count,
    #    workers=workers,
    #    sample=1e-5,
    #    negative=negative,
    #    iter=epochs
    #)

    w2v_model = gensim.models.Word2Vec(
        size=size,
        window=window,
        min_count=min_count,
        workers=workers,
        sample=1e-5,
        negative=negative,
        iter=epochs
    )
    print(w2v_model.layer1_size)

    sentences = gensim.models.word2vec.PathLineSentences(path)

    print("building the vocabulary...")
    w2v_model.build_vocab(sentences)

    print("training the model...")
    w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=w2v_model.iter)

    print("creating the sense2vec model...")
    vector_map = VectorMap(size)

    for string in w2v_model.wv.vocab:
        vocab = w2v_model.wv.vocab[string]
        freq, idx = vocab.count, vocab.index
        if freq < min_count:
            continue
        vector = w2v_model.wv.vectors[idx]
        vector_map.borrow(string, freq, vector)

    print("saving the model to file...")
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    vector_map.save(out_path)


if __name__ == '__main__':
    plac.call(main)
