from tokenizers import ByteLevelBPETokenizer, BertWordPieceTokenizer, CharBPETokenizer, SentencePieceBPETokenizer
from tokenizers import processors
from .params import min_frequences, vocab_sizes, tokenizers
import os
import IPython
from transformers import RobertaTokenizerFast, DataCollatorForLanguageModeling
from tokenizers.processors import BertProcessing



class AmTokenizer(object):
    def __init__(self, vocab_size = 25_000, min_frequence=5, tokenizer_name = "byte_bpe", max_length= None):
        assert vocab_size in vocab_sizes, "Model for vocabulary size:{} is not available!\nModels are avilable for vocab size:{}".format(vocab_size, list(vocab_sizes))
        assert min_frequence in min_frequences, "Model for min frequency:{} is not available!\nModels are avilable for min frequencies:{}".format(min_frequence, list(min_frequences))
        assert tokenizer_name in tokenizers, "Model for tokenizer:{} is not available!\nModels are avilable for tokenizers:{}".format(tokenizer_name, list(tokenizers))
        self.max_length = max_length
        self.__vocab_size = vocab_size
        self.__min_frequence = min_frequence
        self.__tokenizer_name = tokenizer_name
        self.__load_tokenizer()
        
    def __load_tokenizer(self):
        parent_path, _ = os.path.split(__file__)
        data_path = os.path.join(parent_path, "data")
        tokenizer_path = os.path.join(data_path, str(self.__vocab_size), str(self.__min_frequence), self.__tokenizer_name)
    
        self.__tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_path, max_len=self.max_length)
            
        self.__tokenizer._tokenizer.post_processor = BertProcessing(
            ("</s>", self.__tokenizer.convert_tokens_to_ids("</s>")),
            ("<s>", self.__tokenizer.convert_tokens_to_ids("<s>")),
        )

        
                
 
        
        
    def encode(self, string):
        if self.max_length is not None:
            encoded = self.__tokenizer(string, padding="max_length", truncation=True )
        else:
            encoded = self.__tokenizer(string)
       
        return encoded
        
    
    def decode(self, ids):
        return self.__tokenizer.decode(ids)