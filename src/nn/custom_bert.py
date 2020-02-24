import logging

from flair.embeddings import BertEmbeddings, TokenEmbeddings
from transformers import (
    BertTokenizer,
    BertModel,
)

log = logging.getLogger("flair")


class CustomBertEmbeddings(BertEmbeddings):
    def __init__(
        self,
        bert_model_or_path: str = "bert-base-uncased",
        layers: str = "-1,-2,-3,-4",
        pooling_operation: str = "first",
        use_scalar_mix: bool = False,
        from_tf=False,
    ):
        """
        Bidirectional transformer embeddings of words, as proposed in Devlin et al., 2018.
        :param bert_model_or_path: name of BERT model ('') or directory path containing custom model, configuration file
        and vocab file (names of three files should be - config.json, pytorch_model.bin/model.chkpt, vocab.txt)
        :param layers: string indicating which layers to take for embedding
        :param pooling_operation: how to get from token piece embeddings to token embedding. Either pool them and take
        the average ('mean') or use first word piece embedding as token embedding ('first)
        """
        print('custom init')
        super(BertEmbeddings, self).__init__()
        print('initialized')
        if bert_model_or_path.startswith("distilbert"):
            try:
                from transformers import DistilBertTokenizer, DistilBertModel
            except ImportError:
                log.warning("-" * 100)
                log.warning(
                    "ATTENTION! To use DistilBert, please first install a recent version of transformers!"
                )
                log.warning("-" * 100)
                pass

            self.tokenizer = DistilBertTokenizer.from_pretrained(bert_model_or_path, from_tf=from_tf,)
            self.model = DistilBertModel.from_pretrained(
                pretrained_model_name_or_path=bert_model_or_path,
                output_hidden_states=True,
                from_tf=from_tf,
            )
        else:
            self.tokenizer = BertTokenizer.from_pretrained(bert_model_or_path, from_tf=from_tf,)
            self.model = BertModel.from_pretrained(
                pretrained_model_name_or_path=bert_model_or_path,
                output_hidden_states=True,
                from_tf=from_tf,

            )
        self.layer_indexes = [int(x) for x in layers.split(",")]
        self.pooling_operation = pooling_operation
        self.use_scalar_mix = use_scalar_mix
        self.name = str(bert_model_or_path)
        self.static_embeddings = True
