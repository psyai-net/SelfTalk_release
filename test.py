import numpy as np
import argparse
import os
import torch
import time
from data_loader import get_dataloaders
from SelfTalk import SelfTalk


@torch.no_grad()
def test(args, model, test_loader, epoch):
    result_path = os.path.join(args.dataset, args.result_path)
    result_path = result_path + '_' + str(args.feature_dim) + '_' + str(time.strftime("%m_%d_%H_%M", time.localtime()))
    os.makedirs(result_path, exist_ok=True)

    model.load_state_dict(
        torch.load(os.path.join(args.save_path, '{}_model.pth'.format(epoch)), map_location=torch.device('cpu')))
    model = model.to(args.device)
    model.eval()

    for audio, vertice, template, file_name in test_loader:
        # to gpu
        audio, vertice, template = audio.to(args.device), vertice.to(args.device), template.to(args.device)
        prediction, lip_features, logits = model.predict(audio, template)
        prediction = prediction.squeeze()  # (seq_len, V*3)
        prediction = prediction.reshape(prediction.shape[0], -1, 3)
        np.save(os.path.join(result_path, file_name[0].split(".")[0] + ".npy"), prediction.detach().cpu().numpy())


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    # VOCASET
    parser = argparse.ArgumentParser(
        description='SelfTalk: A Self-Supervised Commutative Training Diagram to Comprehend 3D Talking Faces')
    parser.add_argument("--lr", type=float, default=0.0001, help='learning rate')
    parser.add_argument("--dataset", type=str, default="vocaset", help='vocaset or BIWI')
    parser.add_argument("--vertice_dim", type=int, default=5023 * 3,
                        help='number of vertices - 5023*3 for vocaset; 23370*3 for BIWI')
    parser.add_argument("--feature_dim", type=int, default=512, help='512 for vocaset; 1024 for BIWI')
    parser.add_argument("--period", type=int, default=30, help='period in PPE - 30 for vocaset; 25 for BIWI')
    parser.add_argument("--wav_path", type=str, default="wav", help='path of the audio signals')
    parser.add_argument("--vertices_path", type=str, default="vertices_npy", help='path of the ground truth')
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help='gradient accumulation')
    parser.add_argument("--max_epoch", type=int, default=100, help='number of epochs')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--template_file", type=str, default="templates.pkl", help='path of the personalized templates')
    parser.add_argument("--save_path", type=str, default="save", help='path of the trained models')
    parser.add_argument("--result_path", type=str, default="result", help='path to the predictions')
    parser.add_argument("--train_subjects", type=str, default="FaceTalk_170728_03272_TA"
                                                              " FaceTalk_170904_00128_TA FaceTalk_170725_00137_TA FaceTalk_170915_00223_TA"
                                                              " FaceTalk_170811_03274_TA FaceTalk_170913_03279_TA"
                                                              " FaceTalk_170904_03276_TA FaceTalk_170912_03278_TA")
    parser.add_argument("--val_subjects", type=str, default="FaceTalk_170811_03275_TA"
                                                            " FaceTalk_170908_03277_TA")
    parser.add_argument("--test_subjects", type=str, default="FaceTalk_170809_00138_TA"
                                                             " FaceTalk_170731_00024_TA")
    args = parser.parse_args()

    # BIWI
    # parser = argparse.ArgumentParser(description='SelfTalk: A Self-Supervised Commutative Training Diagram to Comprehend 3D Talking Faces')
    # parser.add_argument("--lr", type=float, default=0.0001, help='learning rate')
    # parser.add_argument("--dataset", type=str, default="BIWI", help='vocaset or BIWI')
    # parser.add_argument("--vertice_dim", type=int, default=23370*3, help='number of vertices - 5023*3 for vocaset; 23370*3 for BIWI')
    # parser.add_argument("--feature_dim", type=int, default=1024, help='512 for vocaset; 1024 for BIWI')
    # parser.add_argument("--period", type=int, default=25, help='period in PPE - 30 for vocaset; 25 for BIWI')
    # parser.add_argument("--wav_path", type=str, default= "wav", help='path of the audio signals')
    # parser.add_argument("--vertices_path", type=str, default="vertices_npy", help='path of the ground truth')
    # parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help='gradient accumulation')
    # parser.add_argument("--max_epoch", type=int, default=100, help='number of epochs')
    # parser.add_argument("--device", type=str, default="cuda")
    # parser.add_argument("--template_file", type=str, default="templates.pkl", help='path of the personalized templates')
    # parser.add_argument("--save_path", type=str, default="save", help='path of the trained models')
    # parser.add_argument("--result_path", type=str, default="result", help='path to the predictions')
    # parser.add_argument("--train_subjects", type=str, default="F2 F3 F4 M3 M4 M5")
    # parser.add_argument("--val_subjects", type=str, default="F2 F3 F4 M3 M4 M5")
    # parser.add_argument("--test_subjects", type=str, default="F1 F5 F6 F7 F8 M1 M2 M6")
    # args = parser.parse_args()

    # build model
    model = SelfTalk(args)
    print("model parameters: ", count_parameters(model))

    # to cuda
    assert torch.cuda.is_available()
    model = model.to(args.device)

    # load data
    dataset = get_dataloaders(args)
    test(args, model, dataset["test"], epoch=args.max_epoch)


if __name__ == "__main__":
    main()
