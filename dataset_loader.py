"""
Load AG News and 20 Newsgroups datasets for text classification.
"""

from datasets import load_dataset as load_hf_dataset
from sklearn.datasets import fetch_20newsgroups


def load_ag_news(split=None):
    """
    Load AG News dataset from Hugging Face.
    Classes: World (0), Sports (1), Business (2), Sci/Tech (3).
    Returns Dataset or DatasetDict with 'text' and 'label' columns.
    """
    dataset = load_hf_dataset("ag_news", trust_remote_code=True)
    if split:
        return dataset[split]
    return dataset


def load_20_newsgroups(subset="train", remove=("headers", "footers", "quotes"), categories=None):
    """
    Load 20 Newsgroups dataset via scikit-learn.
    subset: 'train', 'test', or 'all'
    remove: tuple of 'headers', 'footers', 'quotes' to strip from each document
    categories: list of category names, or None for all 20 categories.
    Returns (data, target): lists of text and integer labels.
    """
    data = fetch_20newsgroups(
        subset=subset,
        remove=remove,
        categories=categories,
        shuffle=True,
        random_state=42,
    )
    return data.data, data.target, data.target_names


if __name__ == "__main__":
    print("Loading AG News...")
    ag = load_ag_news()
    print("AG News:", ag)
    print("  Train size:", len(ag["train"]))
    print("  Test size:", len(ag["test"]))
    print("  Example:", ag["train"][0])

    print("\nLoading 20 Newsgroups (train)...")
    texts, labels, class_names = load_20_newsgroups(subset="train")
    print("  20 Newsgroups train size:", len(texts))
    print("  Num classes:", len(class_names))
    print("  Example label:", class_names[labels[0]])
    print("  Example text (first 200 chars):", texts[0][:200] + "...")
