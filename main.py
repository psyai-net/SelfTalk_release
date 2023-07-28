import numpy as np
import argparse
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2Processor
import torch
import torch.nn as nn
import time
from data_loader import get_dataloaders
from SelfTalk import SelfTalk


def trainer(args, train_loader, dev_loader, model, optimizer, criterion, epoch, last_train):
    save_path = os.path.join(args.dataset, args.save_path)
    save_path = save_path + '_' + str(args.feature_dim) + '_' + str(time.strftime("%m_%d_%H_%M", time.localtime()))
    os.makedirs(save_path, exist_ok=True)
    if last_train != 0:
        model.load_state_dict(torch.load(os.path.join(args.load_path, '{}_model.pth'.format(last_train)),
                                         map_location=torch.device('cpu')))
        model = model.to(args.device)
    writer = SummaryWriter(log_dir="log/{}".format(save_path.split('/')[-1]))
    processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")
    iteration = 0
    for e in range(epoch + 1):
        loss_log = []
        # train
        model.train()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        optimizer.zero_grad()

        for i, (audio, vertice, template, file_name) in pbar:
            iteration += 1
            # to gpu
            audio, vertice, template = audio.to(args.device), vertice.to(args.device), template.to(args.device)
            vertice_out, vertice, lip_features, text_hidden_states, logits, text_logits = model(audio, template,
                                                                                                vertice)
            loss1 = criterion(vertice_out, vertice)
            gt_vel = vertice[:, 1:, :] - vertice[:, :-1, :]
            pred_vel = vertice_out[:, 1:, :] - vertice_out[:, :-1, :]
            loss2 = criterion(pred_vel, gt_vel)
            loss3 = criterion(lip_features, text_hidden_states)
            text_logits = torch.argmax(text_logits, dim=-1)
            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)
            text_logits = processor.batch_decode(text_logits)
            text_logits = tokenizer(text_logits, return_tensors="pt").input_ids
            text_logits = text_logits.to(args.device)
            loss4 = nn.functional.ctc_loss(
                log_probs,
                text_logits,
                torch.tensor([log_probs.shape[0]]),
                torch.tensor([text_logits.shape[1]]),
                blank=0,
                reduction="mean",
                zero_infinity=True,
            )
            loss = torch.mean(1000 * loss1 + 1000 * loss2 + 0.001 * loss3 + 0.0001 * loss4)
            loss.backward()
            loss_log.append(loss.item())
            if i % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            pbar.set_description(
                "(Epoch {}, iteration {}) TRAIN LOSS:{:.7f}, loss1:{:.7f}, loss2:{:.7f}, loss3:{:.7f}, loss4:{:.7f}".format(
                    e, iteration, loss.item(), loss1.item(), loss2.item(), loss3.item(), loss4.item()))
        writer.add_scalar("train/loss1", loss1.item(), e)
        writer.add_scalar("train/loss2", loss2.item(), e)
        writer.add_scalar("train/loss3", loss3.item(), e)
        writer.add_scalar("train/loss4", loss4.item(), e)
        writer.add_scalar("train/loss", loss.item(), e)
        writer.add_scalar("train/lr", optimizer.param_groups[0]['lr'], e)

        # validation
        valid_loss_log = []
        model.eval()
        with torch.no_grad():

            for audio, vertice, template, file_name in dev_loader:
                # to gpu
                audio, vertice, template = audio.to(args.device), vertice.to(args.device), template.to(args.device)
                train_subject = "_".join(file_name[0].split("_")[:-1])
                vertice_out, vertice, lip_features, text_hidden_states, logits, text_logits = model(audio, template,
                                                                                                    vertice)
                loss = criterion(vertice_out, vertice)
                valid_loss_log.append(loss.item())
            writer.add_scalar("val_loss", loss.item(), e)

        current_loss = np.mean(valid_loss_log)

        if (e > 0 and e % 25 == 0) or e == args.max_epoch:
            torch.save(model.state_dict(), os.path.join(save_path, '{}_model.pth'.format(e)))

        print("epcoh: {}, current loss:{:.7f}".format(e + 1, current_loss))
    writer.close()
    return model


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
    parser.add_argument("--device", type=str, default="cuda", help='cuda or cpu')
    parser.add_argument("--last_train", type=int, default=0, help='last train')
    parser.add_argument("--load_path", type=str, default=None, help='path to the trained models')
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
    # parser.add_argument("--max_epoch", type=int, default=1000, help='number of epochs')
    # parser.add_argument("--device", type=str, default="cuda", help='cuda or cpu')
    # parser.add_argument("--template_file", type=str, default="templates.pkl", help='path of the personalized templates')
    # parser.add_argument("--save_path", type=str, default="save", help='path of the trained models')
    # parser.add_argument("--result_path", type=str, default="result", help='path to the predictions')
    # parser.add_argument("--last_train", type=int, default=0, help='last train')
    # parser.add_argument("--load_path", type=str, default=None, help='path to the trained models')
    # parser.add_argument("--train_subjects", type=str, default="F2 F3 F4 M3 M4 M5")
    # parser.add_argument("--val_subjects", type=str, default="F2 F3 F4 M3 M4 M5")
    # parser.add_argument("--test_subjects", type=str, default="F2 F3 F4 M3 M4 M5")
    # args = parser.parse_args()

    model = SelfTalk(args)
    print("model parameters: ", count_parameters(model))

    # to cuda
    assert torch.cuda.is_available()
    model = model.to(args.device)

    # load data
    dataset = get_dataloaders(args)
    # loss
    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    trainer(args, dataset["train"], dataset["valid"], model, optimizer, criterion, epoch=args.max_epoch,
            last_train=args.last_train)


if __name__ == "__main__":
    main()
