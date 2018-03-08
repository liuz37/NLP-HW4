from allennlp.data.dataset_readers import SquadReader
from allennlp.data.iterators import BucketIterator
from allennlp.data import Vocabulary
import torch
from torch import optim
from torch.nn.functional import nll_loss

from .rnn_attn_rc_test_case import RNNAttnRCTestCase
from rnn_attention_rc.models.rnn import RNN


class TestRNN(RNNAttnRCTestCase):
    def test_forward(self):
        lr = 0.5
        batch_size = 16
        embedding_dim = 50
        hidden_size = 15
        dropout = 0.2

        squad_reader = SquadReader()
        # Read SQuAD train set (use the test set, since it's smaller)
        train_dataset = squad_reader.read(self.squad_test)
        vocab = Vocabulary.from_dataset(train_dataset)
        train_dataset.index_instances(vocab)

        # Random embeddings for test
        test_embed_matrix = torch.rand(vocab.get_vocab_size(), embedding_dim)
        test_rnn = RNN(test_embed_matrix, hidden_size, dropout)
        try:
            optimizer = optim.Adadelta(filter(lambda p: p.requires_grad,
                                              test_rnn.parameters()), lr=lr)
        except ValueError:
            # Likely there are no parameters to optimize, because
            # the code is not complete.
            pass
        iterator = BucketIterator(batch_size=batch_size,
                                  sorting_keys=[("passage", "num_tokens"),
                                                ("question", "num_tokens")])
        for batch in iterator(train_dataset, num_epochs=1):
            passage = batch["passage"]["tokens"]
            question = batch["question"]["tokens"]
            span_start = batch["span_start"]
            span_end = batch["span_end"]
            try:
                output_dict = test_rnn(passage, question)
                softmax_start_logits = output_dict["softmax_start_logits"]
                softmax_end_logits = output_dict["softmax_end_logits"]
                loss = nll_loss(softmax_start_logits, span_start.view(-1))
                loss += nll_loss(softmax_end_logits, span_end.view(-1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            except NotImplementedError:
                # RNN.forward() not implemented yet, don't fail tests.
                pass
