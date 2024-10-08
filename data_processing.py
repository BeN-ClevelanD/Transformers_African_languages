import torch
from torch.utils.data import Dataset, DataLoader
import os


#Class that tokenizes the text at a character level
class CharacterTokenizer:
    """
    A simple character-level tokenizer.
    
    This tokenizer creates a mapping between characters and integer indices,
    including special tokens for unknown characters and end-of-line.
    """
    def __init__(self, text: str):
        """
        Initialize the tokenizer with the given text.
        
        Args:
            text (str): The text to use for building the vocabulary.
        """
        unique_chars = sorted(set(text))
        #Builds a dictionary that maps characters to indices and vice versa
        self.char_to_idx = {'<UNK>': 0, '<ELO>': 1}
        for i, char in enumerate(unique_chars, start=2):
            self.char_to_idx[char] = i
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)
    #Encodes the text/characters/tokens into a list of indices
    def encode(self, text: str) -> list[int]:
        """
        Encode the given text into a list of integer indices.
        
        Args:
            text (str): The text to encode.
        
        Returns:
            list[int]: The encoded text as a list of integer indices.
        """
        return [self.char_to_idx.get(char, self.char_to_idx['<UNK>']) for char in text]
    #Decodes the list of indices into text, or characters/tokens
    def decode(self, indices: list[int]) -> str:
        """
        Decode the given list of integer indices back into text.
        
        Args:
            indices (list[int]): The list of integer indices to decode.
        
        Returns:
            str: The decoded text.
        """
        return ''.join([self.idx_to_char[idx] for idx in indices])
    
#Class that creates a dataset from the text data
class TextDataset(Dataset):
    """
    A PyTorch Dataset for character-level language modeling.
    """
    def __init__(self, text: str, seq_length: int, tokenizer: CharacterTokenizer):
        """
        Initialize the dataset.
        
        Args:
            text (str): The full text to use for the dataset.
            seq_length (int): The length of sequences to generate.
            tokenizer (CharacterTokenizer): The tokenizer to use for encoding the text.
        """
        self.text = text
        self.seq_length = seq_length
        self.tokenizer = tokenizer
        #Encodes the text into a list of indices using the passed in tokenizer
        self.data = self.tokenizer.encode(self.text)
    #Returns the length of the dataset
    def __len__(self) -> int:
        """
        Get the length of the dataset.
        
        Returns:
            int: The number of sequences in the dataset.
        """
        return max(0, len(self.data) - self.seq_length)
    #Function that creates input and target pairs from the data for our model to train on 
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single item from the dataset.
        
        Args:
            idx (int): The index of the item to retrieve.
        
        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing the input sequence and target sequence.
        """
        inputs = torch.tensor(self.data[idx:idx+self.seq_length])
        #Displaced the target by one position to the right
        targets = torch.tensor(self.data[idx+1:idx+self.seq_length+1])
        return inputs, targets
#Function that loads the data from the file path
def load_data(file_path: str) -> str:
    """
    Load text data from a file, replacing newlines with a special token.
    
    Args:
        file_path (str): The path to the file to load.
    
    Returns:
        str: The loaded text with newlines replaced by '<ELO>'.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        # Replace newlines with <EOL> token
        return f.read().replace('\n', '<ELO>')
#Function that enacts all of the main preprocessing steps for the data partitions
def load_and_preprocess_data(data_dir: str, seq_length: int) -> tuple[TextDataset, TextDataset, TextDataset, CharacterTokenizer]:
    """
    Load and preprocess data from train, validation, and test files.
    
    Args:
        data_dir (str): The directory containing the data files.
        seq_length (int): The length of sequences to use in the datasets.
    
    Returns:
        tuple[TextDataset, TextDataset, TextDataset, CharacterTokenizer]: 
        A tuple containing the train, validation, and test datasets, and the tokenizer.
    """
    train_path = os.path.join(data_dir, 'nchlt_text.nr.train')
    valid_path = os.path.join(data_dir, 'nchlt_text.nr.valid')
    test_path = os.path.join(data_dir, 'nchlt_text.nr.test')

    train_text = load_data(train_path)
    valid_text = load_data(valid_path)
    test_text = load_data(test_path)
    #Creates a tokenizer object from the training text
    tokenizer = CharacterTokenizer(train_text)
    #Creates the datasets from the text data, tokenizes them using the above tokenizer
    train_dataset = TextDataset(train_text, seq_length, tokenizer)
    valid_dataset = TextDataset(valid_text, seq_length, tokenizer)
    test_dataset = TextDataset(test_text, seq_length, tokenizer)

    return train_dataset, valid_dataset, test_dataset, tokenizer
#Function that creates the torch data loaders for the training, validation and test datasets
def get_data_loaders(train_dataset: TextDataset, valid_dataset: TextDataset, test_dataset: TextDataset, batch_size: int) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoader objects for train, validation, and test datasets.
    
    Args:
        train_dataset (TextDataset): The training dataset.
        valid_dataset (TextDataset): The validation dataset.
        test_dataset (TextDataset): The test dataset.
        batch_size (int): The batch size to use for the DataLoaders.
    
    Returns:
        tuple[DataLoader, DataLoader, DataLoader]: 
        A tuple containing the train, validation, and test DataLoaders.
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, valid_loader, test_loader