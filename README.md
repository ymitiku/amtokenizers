
# Amharic Language Tokenizers

This package contains set of Classes which can be used to encode Amharic language sentences into tokens that could be used by language models. The tokenizers are trained using [Contemporary Amharic Corpus (CACO)](https://www.findke.ovgu.de/findke/en/Research/Data+Sets/Contemporary+Amharic+Corpus+%28CACO%29-p-1142.html) dataset


## Installing


#### Pip installation

```bash
pip install -i https://test.pypi.org/simple/ amtokenizers==0.0.5
```

## Sample Code

### Variable length 
```python
from amtokenizers import AmTokenizer

a  = AmTokenizer(10000, 5 , "byte_bpe")
encoded = a.encode("አበበ በሶ በላ።", return_tokens=False)
print("encoded", encoded.tokens)
# encoded ['<s>', 'áĬł', 'áīłáīł', 'ĠáīłáĪ¶', 'ĠáīłáĪĭ', 'áį', '¢', '</s>']
print("decoded:", a.decode(encoded.ids))
# decoded: <s>አበበ በሶ በላ።</s>
```
### Fixed length

```python
a  = AmTokenizer(10000, 5 , "byte_bpe", max_length=16)
encoded = a.encode("አበበ በሶ በላ።")
print("encoded", encoded.tokens())
# encoded ['<s>', 'áĬł', 'áīłáīł', 'ĠáīłáĪ¶', 'ĠáīłáĪĭ', 'áį', '¢', '</s>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>']
print(encoded.input_ids)
# [0, 337, 3251, 3598, 3486, 270, 100, 2, 1, 1, 1, 1, 1, 1, 1, 1]
print("decoded:", a.decode(encoded.input_ids))
# decoded: <s>አበበ በሶ በላ።</s><pad><pad><pad><pad><pad><pad><pad><pad>
```

<h2>Disclaimer</h2>

This package is highly inspired by Hugging Face's [How to train a new language model from scratch using Transformers and Tokenizers](https://huggingface.co/blog/how-to-train) tutorial.

