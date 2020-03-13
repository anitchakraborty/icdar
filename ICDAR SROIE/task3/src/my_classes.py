import numpy


class TextBox(object):
    """TextBox object with x, y, x_span and y_span, and the text"""
    def __init__(self, line):
        line_split = line.strip().split(",", maxsplit=8)
        self.x_span = (int(line_split[0]), int(line_split[4]))
        self.y_span = (int(line_split[1]), int(line_split[5]))
        self.x = (self.x_span[0] + self.xspan[1]) / 2
        self.y = (self.y_span[0] + self.y_span[1]) / 2
        self.text = line_split[8]

    def __repr__(self):
        return self.text


class TextLine(object):
    """TestLine class: Facilitates inserting texts inside a TextBox object, if permissible
       Method(s): TextLine.().insert(textbox)
    """
    def __init__(self, textbox=None):
        if isinstance(textbox, TextBox):
            self.text = [textbox.text]
            self.xs = [textbox.x]
            self.y = textbox.y
            self.y_span = textbox.y_span
        else:
            self.text = []
            self.xs = []
            self.y_span = None

    def insert(self, textbox):
        if not (
            (textbox.y_span[0] < self.y < textbox.y_span[1])
            and (self.y_span[0] < textbox.y < self.y_span[1])
        ):
            raise ValueError

        try:
            at = next(i for i, v in enumerate(self.xs) if v > textbox.x)
            self.text.insert(at, textbox.text)
            self.xs.insert(at, textbox.x)
        except StopIteration:
            self.text.append(textbox.text)
            self.xs.append(textbox.x)

        self.y = textbox.y
        self.y_span = textbox.y_span

    def __str__(self):
        return "\t".join(self.text)

    def __repr__(self):
        if self.y_span is None:
            repr_y_span = "[    ,    ] "
        else:
            repr_y_span = "[{:4d},{:4d}] ".format(self.y_span[0], self.y_span[1])

        repr_text = "\t".join(self.text)

        return repr_y_span + repr_text


if __name__ == "__main__":
    pass
