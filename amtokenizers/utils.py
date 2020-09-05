from tokenizers import ByteLevelBPETokenizer, BertWordPieceTokenizer, CharBPETokenizer, SentencePieceBPETokenizer
from tokenizers import processors
from .params import min_frequences, vocab_sizes, tokenizers
import os


class AmTokenizer(object):
    def __init__(self, vocab_size = 25_000, min_frequence=5, tokenizer_name = "byte_bpe"):
        assert vocab_size in vocab_sizes, "Model for vocabulary size:{} is not available!\nModels are avilable for vocab size:{}".format(vocab_size, list(vocab_sizes))
        assert min_frequence in min_frequences, "Model for min frequency:{} is not available!\nModels are avilable for min frequencies:{}".format(min_frequence, list(min_frequences))
        assert tokenizer_name in tokenizers, "Model for tokenizer:{} is not available!\nModels are avilable for tokenizers:{}".format(tokenizer_name, list(tokenizers))
        
        self.__vocab_size = vocab_size
        self.__min_frequence = min_frequence
        self.__tokenizer_name = tokenizer_name
        self.__load_tokenizer()
    def __load_tokenizer(self):
        parent_path, _ = os.path.split(__file__)

        
        data_path = os.path.join(parent_path, "data")
        merge_file_path = os.path.join(data_path, "amhtok-{}-{}-{}-merges.txt".format(self.__vocab_size, self.__min_frequence, self.__tokenizer_name))
        vocab_file_path = os.path.join(data_path, "amhtok-{}-{}-{}-vocab.json".format(self.__vocab_size, self.__min_frequence, self.__tokenizer_name))
        
        self.__tokenizer = self.__get_tokenizer(self.__tokenizer_name)(vocab_file_path, merge_file_path)
        self.__tokenizer._tokenizer.post_processor = processors.BertProcessing(
            ("</s>", self.__tokenizer.token_to_id("</s>")),
            ("<s>", self.__tokenizer.token_to_id("<s>")),
        )
        self.__tokenizer.enable_truncation(max_length=512)
        
                
    def __get_tokenizer(self, name):
        if name == "word_peice":
            return BertWordPieceTokenizer
        elif name == "byte_bpe":
            return ByteLevelBPETokenizer
        elif name == "char_bpe":
            return CharBPETokenizer
        else:
            return SentencePieceBPETokenizer
        
    def __get_precessor(self, name):
        return BertProcessing
        
        
    def encode(self, string, return_tokens = True):
        
        encoded = self.__tokenizer.encode(string)
        print(encoded)
        if return_tokens:
            encoded = encoded.tokens
        return encoded
        
    
    def decode(self, ids):
        return self.__tokenizer.decode(ids)