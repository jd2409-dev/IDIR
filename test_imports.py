import torch
import torch.nn as nn
import torch.nn.functional as F
from idir_model import IDIR
from teacher_model import TeacherModel

try:
    from datasets import load_dataset
    from transformers import GPT2Tokenizer
except ImportError:
    print("Install datasets and transformers: pip install datasets transformers")
    raise

print("All imports successful")
