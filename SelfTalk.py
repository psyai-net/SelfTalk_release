import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from wav2vec import Wav2Vec2Model, Wav2Vec2ForCTC, linear_interpolation


# Temporal Bias, brrowed from https://github.com/EvelynFan/FaceFormer/blob/main/faceformer.py
def init_biased_mask(n_head, max_seq_len, period):
    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = (2 ** (-2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio ** i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2 * closest_power_of_2)[0::2][
                                                               :n - closest_power_of_2]

    slopes = torch.Tensor(get_slopes(n_head))
    bias = torch.arange(start=0, end=max_seq_len, step=period).unsqueeze(1).repeat(1, period).view(-1) // (period)
    bias = - torch.flip(bias, dims=[0])
    alibi = torch.zeros(max_seq_len, max_seq_len)
    for i in range(max_seq_len):
        alibi[i, :i + 1] = bias[-(i + 1):]
    alibi = slopes.unsqueeze(1).unsqueeze(1) * alibi.unsqueeze(0)
    mask = (torch.triu(torch.ones(max_seq_len, max_seq_len)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    mask = mask.unsqueeze(0) + alibi
    return mask


# Input Representation Adjustment, brrowed from https://github.com/galib360/FaceXHuBERT
def inputRepresentationAdjustment(audio_embedding_matrix, vertex_matrix, ifps, ofps):
    if ifps % ofps == 0:
        factor = -1 * (-ifps // ofps)
        if audio_embedding_matrix.shape[1] % 2 != 0:
            audio_embedding_matrix = audio_embedding_matrix[:, :audio_embedding_matrix.shape[1] - 1]

        if audio_embedding_matrix.shape[1] > vertex_matrix.shape[1] * 2:
            audio_embedding_matrix = audio_embedding_matrix[:, :vertex_matrix.shape[1] * 2]

        elif audio_embedding_matrix.shape[1] < vertex_matrix.shape[1] * 2:
            vertex_matrix = vertex_matrix[:, :audio_embedding_matrix.shape[1] // 2]
    else:
        factor = -1 * (-ifps // ofps)
        audio_embedding_seq_len = vertex_matrix.shape[1] * factor
        audio_embedding_matrix = audio_embedding_matrix.transpose(1, 2)
        audio_embedding_matrix = F.interpolate(audio_embedding_matrix, size=audio_embedding_seq_len, align_corners=True,
                                               mode='linear')
        audio_embedding_matrix = audio_embedding_matrix.transpose(1, 2)

    frame_num = vertex_matrix.shape[1]
    audio_embedding_matrix = torch.reshape(audio_embedding_matrix, (
    1, audio_embedding_matrix.shape[1] // factor, audio_embedding_matrix.shape[2] * factor))
    return audio_embedding_matrix, vertex_matrix, frame_num


class SelfTalk(nn.Module):
    def __init__(self, args):
        super(SelfTalk, self).__init__()
        self.dataset = args.dataset
        self.audio_encoder = Wav2Vec2Model.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")
        self.text_encoder = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        self.audio_encoder.feature_extractor._freeze_parameters()

        if args.dataset == "vocaset":
            pkl_path = "./vocaset/FLAME_masks.pkl"
            with open(pkl_path, 'rb') as f:
                self.lip_mask = pickle.load(f, encoding='latin1')["lips"]
                self.lip_map = nn.Linear(254 * 3, 1024)

        elif args.dataset == "BIWI":
            with open('./BIWI/BIWI_lip.pkl', 'rb') as f:
                self.lip_mask = pickle.load(f, encoding='latin1')
                self.lip_map = nn.Linear(4996 * 3, 1024)

        self.biased_mask = init_biased_mask(n_head=4, max_seq_len=600, period=args.period)
        decoder_layer = nn.TransformerDecoderLayer(d_model=args.feature_dim, nhead=4,
                                                   dim_feedforward=2 * args.feature_dim, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)
        if args.dataset == "vocaset":
            self.audio_feature_map = nn.Linear(1024, args.feature_dim)
            self.transformer = nn.Transformer(d_model=1024, batch_first=True)
        elif args.dataset == "BIWI":
            self.audio_feature_map = nn.Linear(2048, args.feature_dim)
            self.transformer = nn.Transformer(d_model=1024, batch_first=True)
        self.vertice_map_r = nn.Linear(args.feature_dim, args.vertice_dim)
        self.device = args.device
        self.dropout = nn.Dropout(p=0.0, inplace=False)
        self.lm_head = nn.Linear(1024, 33)

        nn.init.constant_(self.vertice_map_r.weight, 0)
        nn.init.constant_(self.vertice_map_r.bias, 0)

    def forward(self, audio, template, vertice):

        template = template.unsqueeze(1)
        frame_num = vertice.shape[1]
        hidden_states = self.audio_encoder(audio, self.dataset, frame_num=frame_num).last_hidden_state

        if self.dataset == "BIWI":
            hidden_states, vertice, frame_num = inputRepresentationAdjustment(hidden_states, vertice, 50, 25)
            hidden_states = hidden_states[:, :frame_num]
        elif self.dataset == "vocaset":
            pass
        vertice_input = self.audio_feature_map(hidden_states)
        vertice_out = self.transformer_decoder(vertice_input, vertice_input)
        vertice_out = self.vertice_map_r(vertice_out)
        audio_model = self.text_encoder(audio)
        text_hidden_states = audio_model.hidden_states
        text_logits = audio_model.logits
        frame_num = text_hidden_states.shape[1]
        lip_out = vertice_out.reshape(vertice_out.shape[0], vertice_out.shape[1], -1, 3)[:, :, self.lip_mask,
                  :].reshape(vertice_out.shape[0], vertice_out.shape[1], -1)
        lip_gt = vertice.reshape(vertice.shape[0], vertice.shape[1], -1, 3)[:, :, self.lip_mask, :].reshape(
            vertice.shape[0], vertice.shape[1], -1)
        lip_offset = self.lip_map(lip_out)

        if self.dataset == "vocaset":
            lip_offset = linear_interpolation(lip_offset, 30, 50, output_len=frame_num)
        elif self.dataset == "BIWI":
            text_hidden_states = text_hidden_states[:, :vertice_out.shape[1] * 2]
            text_logits = text_logits[:, :vertice_out.shape[1] * 2]
            frame_num = text_hidden_states.shape[1]
            lip_offset = linear_interpolation(lip_offset, 25, 50, output_len=frame_num)
        lip_features = self.transformer(lip_offset, lip_offset)
        logits = self.lm_head(self.dropout(lip_features))
        vertice_out = vertice_out + template

        return vertice_out, vertice, lip_features, text_hidden_states, logits, text_logits

    def predict(self, audio, template):
        template = template.unsqueeze(1)
        hidden_states = self.audio_encoder(audio, self.dataset).last_hidden_state
        if self.dataset == "BIWI":
            if hidden_states.shape[1] % 2 != 0:
                hidden_states = hidden_states[:, :hidden_states.shape[1] - 1]
            hidden_states = torch.reshape(hidden_states, (1, hidden_states.shape[1] // 2, hidden_states.shape[2] * 2))
        elif self.dataset == "vocaset":
            pass
        vertice_input = self.audio_feature_map(hidden_states)
        vertice_out = self.transformer_decoder(vertice_input, vertice_input)
        vertice_out = self.vertice_map_r(vertice_out)
        lip_offset = vertice_out.reshape(vertice_out.shape[0], vertice_out.shape[1], -1, 3)[:, :, self.lip_mask,
                     :].reshape(vertice_out.shape[0], vertice_out.shape[1], -1)
        lip_offset = self.lip_map(lip_offset)

        if self.dataset == "vocaset":
            lip_offset = linear_interpolation(lip_offset, 30, 50, output_len=None)
        elif self.dataset == "BIWI":
            lip_offset = linear_interpolation(lip_offset, 25, 50, output_len=None)
        lip_features = self.transformer(lip_offset, lip_offset)
        if self.dataset == "vocaset":
            vertice_out = vertice_out + template
        elif self.dataset == "BIWI":
            vertice_out = vertice_out + template
        logits = self.lm_head(self.dropout(lip_features))

        return vertice_out, lip_features, logits
