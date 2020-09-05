
# Amharic Langugae Tokenizers

This package contains set of Classes which can be used to encode Amharic language sentences into tokens that could be used by language models. The tokenizers are trained using [Contemporary Amharic Corpus (CACO)](https://www.findke.ovgu.de/findke/en/Research/Data+Sets/Contemporary+Amharic+Corpus+%28CACO%29-p-1142.html) dataset


## Installing


#### Pip installation

```bash
pip install -i https://test.pypi.org/simple/ amtokenizers==0.0.5
```

## Sample Code

```python
from amtokenizers import AmTokenizer

a  = AmTokenizer(10000, 5 , "byte_bpe")
encoded = a.encode("አበበ በሶ በላ።", return_tokens=False)
print("encoded", encoded.tokens)

print("decoded:", a.decode(encoded.ids))

```

<h2>Disclaimer</h2>

This package is highly inspired by Hugging Face's [How to train a new language model from scratch using Transformers and Tokenizers](https://huggingface.co/blog/how-to-train) tutorial.

