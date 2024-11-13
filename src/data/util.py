import re
from nltk.tokenize.treebank import TreebankWordDetokenizer as Detok


class Detokenizer(object):
    # https://stackoverflow.com/a/46311499
    def __init__(self) -> None:
        self.detokenizer = Detok()

    def __call__(self, tokens, return_offsets=False):
        text = self.detokenizer.detokenize(tokens)
        text = re.sub('\s*,\s*', ', ', text)
        text = re.sub('\s*\.\s*', '. ', text)
        text = re.sub('\s*\?\s*', '? ', text)
        text = text.replace(" --", "--")

        if return_offsets:
            offsets = [0]
            for i in range(1, len(tokens)):
                offsets.append(len(self(tokens[:i])))

            """
            # verify offsets
            for i, offset in enumerate(offsets):
                if i == 0:
                    continue
                check = text[:offset]
                target = self(tokens[:i])
                try:
                    assert target == check
                except AssertionError:
                    print(tokens)
                    print(f"'{check}' != '{target}'")
                    raise AssertionError
            """

            return text.strip(), offsets
        return text.strip()


