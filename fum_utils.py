import bs4, chardet, nltk, os


def check_extension(fname, extension = ".csv"):
    """
    Checks whether the fname includes an extension.
    Adds an extension if none exists.

    fname - the name of the file to check.
    extension - the extension to append if necessary.
     >> Default: ".csv".
    """

    root, ending = os.path.splitext(fname)
    if not ending:
        ending = extension
    return root + ending


def validate_column_index(index, width, name = "column"):
    """
    Converts negative column indexes and verifies the result is within row bounds.
    """

    if width <= 0:
        raise ValueError("Cannot select a column from an empty row.")

    resolved = width + index if index < 0 else index
    if resolved < 0 or resolved >= width:
        raise IndexError(f"{name} index {index} is out of range for {width} columns.")
    return resolved


def prepare_output_path(fname, extension = ".csv"):
    """
    Validates and normalizes an output file path before writing.
    """

    if not fname:
        raise ValueError("Output file name must not be empty.")
    if os.path.isdir(str(fname)):
        raise IsADirectoryError(f"Output path is a directory: {fname}")

    path = check_extension(str(fname), extension)
    directory = os.path.dirname(os.path.abspath(path))

    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Output directory does not exist: {directory}")
    if not os.access(directory, os.W_OK):
        raise PermissionError(f"Output directory is not writable: {directory}")
    if os.path.isdir(path):
        raise IsADirectoryError(f"Output path is a directory: {path}")
    if os.path.exists(path) and not os.access(path, os.W_OK):
        raise PermissionError(f"Output file is not writable: {path}")

    return path


def clean_documents(x, stop_words = []):
    """
    Removes HTML, stopwords, and words of fewer than three characters from text.
    x - a list of textual documents (strings).
    
    Returns a list of cleaned textual documents (strings).
    """
    
    return tuple(tuple(word for word in \
            nltk.RegexpTokenizer("[a-z]{3,}").tokenize(bs4.BeautifulSoup(document.lower(),
            features = "html.parser").get_text()) if word not in stop_words) for document in x)


def guess_encoding(fname):
    """
    Guesses the encoding of a file.
    fname - the name of the file whose encoding to guess.

    Returns the guessed encoding of the file (string).
    """

    with open(fname, "rb") as file:
        result = chardet.detect(file.read())
    return result["encoding"]
