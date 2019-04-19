# import util
import torch

dataBasePath='/home/lixin/newCourses/11785/hw4p2/data/'


device="cuda" if torch.cuda.is_available() else "cpu"


embedding_size=256
kqv_size=256
listener_hidden_size=256
speller_hidden_size=512
frame_dim=40
batch_size=64