# import util
import torch

dataBasePath='/home/lixin/newCourses/11785/hw4p2/data/'
# dataBasePath='/home/lixin/hw4/data/data/'

device="cuda" if torch.cuda.is_available() else "cpu"


embedding_size=256
kqv_size=128
listener_hidden_size=256
speller_hidden_size=512
frame_dim=40
batch_size=1
output_dim=34
learning_rate=0.001
epoch_num=40
dictionary_length=34
teacher_forcing=0
predict_result="prediction.csv"
beam_width=4
max_generate_length=100
print_cut=150
encoder_dropout=0.1
clip_value=10
is_pretrain=True
