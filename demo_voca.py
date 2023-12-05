import gc
from concurrent.futures import ThreadPoolExecutor
from itertools import cycle, islice

from PIL import Image
import numpy as np
import librosa
import os, argparse, pickle

from pytorch3d.io import load_obj
from pytorch3d.renderer import TexturesVertex, FoVPerspectiveCameras, RasterizationSettings, DirectionalLights, \
    BlendParams, MeshRenderer, MeshRasterizer, SoftPhongShader, look_at_view_transform
from pytorch3d.structures import Meshes
from tqdm import tqdm
import torch.multiprocessing as mp

from SelfTalk import SelfTalk
from transformers import Wav2Vec2Processor
import torch
import time
import cv2
import tempfile
from subprocess import call
from psbody.mesh import Mesh
import trimesh
from loguru import logger


# os.environ['PYOPENGL_PLATFORM'] = 'egl'  # egl


@torch.no_grad()
def test_model(args):
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)

    # build model
    model = SelfTalk(args)
    model.load_state_dict(torch.load(os.path.join(args.dataset, '{}.pth'.format(args.model_name)),
                                     map_location=torch.device(args.device)))
    model = model.to(torch.device(args.device))
    model.eval()

    template_file = os.path.join(args.dataset, args.template_path)
    with open(template_file, 'rb') as fin:
        templates = pickle.load(fin, encoding='latin1')

    temp = templates[args.subject]

    template = temp.reshape((-1))
    template = np.reshape(template, (-1, template.shape[0]))
    template = torch.FloatTensor(template).to(device=args.device)

    wav_path = args.wav_path
    test_name = os.path.basename(wav_path).split(".")[0]
    speech_array, sampling_rate = librosa.load(os.path.join(wav_path), sr=16000)
    processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")
    audio_feature = np.squeeze(processor(speech_array, sampling_rate=16000).input_values)
    audio_feature = np.reshape(audio_feature, (-1, audio_feature.shape[0]))
    audio_feature = torch.FloatTensor(audio_feature).to(device=args.device)

    start = time.time()
    prediction, _, _ = model.predict(audio_feature, template)
    end = time.time()
    print("Model predict time: ", end - start)
    prediction = prediction.squeeze()
    np.save(os.path.join(args.result_path, test_name), prediction.detach().cpu().numpy())

    del prediction
    del model
    gc.collect()
    torch.cuda.empty_cache()


def get_unit_factor(unit):
    if unit == 'mm':
        return 1000.0
    elif unit == 'cm':
        return 100.0
    elif unit == 'm':
        return 1.0
    else:
        raise ValueError('Unit not supported')


def render_obj(obj_file, output_file, device="cuda", background_color=(0, 0, 0)):
    cameras = FoVPerspectiveCameras(fov=30, device=device)
    raster_settings = RasterizationSettings(
        image_size=800,
        faces_per_pixel=1,
        cull_backfaces=True,
        perspective_correct=True,
        max_faces_per_bin=None,
        blur_radius=0.0
    )
    lights = DirectionalLights(
        device=device,
        direction=((0.0, 0.0, 1),),
        ambient_color=((0.18, 0.18, 0.18),),
        diffuse_color=((0.55, 0.55, 0.55),),
        specular_color=((0.05, 0.05, 0.05),),
    )
    blend_params = BlendParams(background_color=background_color)
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(device=device, cameras=cameras, lights=lights, blend_params=blend_params)
    )

    distance = 0.58
    elevation = 0.0
    azimuth = 0.0
    R, T = look_at_view_transform(distance, elevation, azimuth, device=device)

    verts, faces_idx, _ = load_obj(obj_file, device=device, load_textures=False)
    faces = faces_idx.verts_idx
    # Initialize each vertex to be white in color.
    verts_rgb = torch.ones_like(verts)[None]
    textures = TexturesVertex(verts_features=verts_rgb.to(device))

    obj_mesh = Meshes(
        verts=[verts.to(device)],
        faces=[faces.to(device)],
        textures=textures
    )
    image = renderer(meshes_world=obj_mesh, R=R, T=T, bg_col=torch.tensor([0.0, 0.0, 0.0, 0.0]))
    image = image.cpu().numpy().squeeze()[..., :3]
    image = (image * 255).astype(np.uint8)
    pillow_img = Image.fromarray(image)
    pillow_img.save(output_file)


def write_image_task(data):
    i_frame, mesh_path, image_path = data
    obj_file = os.path.join(mesh_path, f"{i_frame:05d}.obj")
    if not os.path.isfile(obj_file):
        logger.error(f"could not find {obj_file}")
        exit(1)
    png_file = os.path.join(image_path, f"{i_frame:05d}.png")
    render_obj(obj_file, png_file)


def render_sequence_meshes(audio_fname, sequence_vertices, template, out_path, uv_template_fname='',
                           texture_img_fname=''):
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    num_frames = sequence_vertices.shape[0]
    image_path = os.path.join(out_path, 'images')
    mesh_path = os.path.join(out_path, 'meshes')
    os.makedirs(image_path, exist_ok=True)
    iterable = zip(range(num_frames), cycle([mesh_path]), cycle([image_path]))
    with mp.Pool(processes=os.cpu_count()) as executor:
        executor.map(write_image_task, islice(iterable, 0, None))

    video_fname = os.path.join(out_path, 'video.mp4')
    cmd = f"ffmpeg -y -r 30 -i {image_path}/%05d.png -i {audio_fname} -pix_fmt yuv420p -qscale 0 {video_fname}".split()
    call(cmd)


def write_obj_task(data):
    i_frame, sequence_vertices, mesh_out_path, template, vt, ft, texture_img_fname = data
    out_fname = os.path.join(mesh_out_path, '%05d.obj' % i_frame)
    out_mesh = Mesh(sequence_vertices, template.f)
    if vt is not None and ft is not None:
        out_mesh.vt, out_mesh.ft = vt, ft
    if os.path.exists(texture_img_fname):
        out_mesh.set_texture_image(texture_img_fname)
    out_mesh.write_obj(out_fname)


def output_sequence_meshes(sequence_vertices, template, out_path, uv_template_fname='', texture_img_fname=''):
    mesh_out_path = os.path.join(out_path, 'meshes')
    if not os.path.exists(mesh_out_path):
        os.makedirs(mesh_out_path)

    if os.path.exists(uv_template_fname):
        uv_template = Mesh(filename=uv_template_fname)
        vt, ft = uv_template.vt, uv_template.ft
    else:
        vt, ft = None, None

    num_frames = sequence_vertices.shape[0]
    iterable = zip(range(num_frames), sequence_vertices, cycle([mesh_out_path]), cycle([template]), cycle([vt]),
                   cycle([ft]), cycle([texture_img_fname]))
    with mp.Pool(processes=os.cpu_count() * 2) as p:
        _ = list(tqdm(p.imap_unordered(write_obj_task, islice(iterable, 0, None)), total=num_frames,
                      desc='output sequence meshes'))


def main():
    parser = argparse.ArgumentParser(
        description='SelfTalk: A Self-Supervised Commutative Training Diagram to Comprehend 3D Talking Faces')
    parser.add_argument("--model_name", type=str, default="vocaset", help='vocaset or BIWI')
    parser.add_argument("--dataset", type=str, default="vocaset", help='vocaset or BIWI')
    parser.add_argument("--fps", type=float, default=30, help='frame rate - 30 for vocaset; 25 for BIWI')
    parser.add_argument("--feature_dim", type=int, default=512, help='512 for vocaset; 1024 for BIWI')
    parser.add_argument("--period", type=int, default=30, help='period in PPE - 30 for vocaset; 25 for BIWI')
    parser.add_argument("--vertice_dim", type=int, default=5023 * 3,
                        help='number of vertices - 5023*3 for vocaset; 23370*3 for BIWI')
    parser.add_argument("--device", type=str, default="cuda", help='cuda or cpu')
    parser.add_argument("--train_subjects", type=str, default="FaceTalk_170728_03272_TA"
                                                              " FaceTalk_170904_00128_TA FaceTalk_170725_00137_TA FaceTalk_170915_00223_TA"
                                                              " FaceTalk_170811_03274_TA FaceTalk_170913_03279_TA"
                                                              " FaceTalk_170904_03276_TA FaceTalk_170912_03278_TA")
    parser.add_argument("--test_subjects", type=str, default="FaceTalk_170809_00138_TA"
                                                             " FaceTalk_170731_00024_TA")
    parser.add_argument("--output_path", type=str, default="demo/output", help='path of the rendered video sequence')
    parser.add_argument("--wav_path", type=str, default="demo/wav/test.wav", help='path of the input audio signal')
    parser.add_argument("--result_path", type=str, default="demo/result", help='path of the predictions')
    parser.add_argument("--subject", type=str, default="FaceTalk_170908_03277_TA",
                        help='select a subject from test_subjects or train_subjects')
    parser.add_argument("--background_black", type=bool, default=True, help='whether to use black background')
    parser.add_argument("--template_path", type=str, default="templates.pkl", help='path of the personalized templates')
    parser.add_argument("--render_template_path", type=str, default="templates",
                        help='path of the mesh in BIWI/FLAME topology')
    args = parser.parse_args()

    test_model(args)
    fa_path = args.result_path + "/" + args.wav_path.split("/")[-1].split(".")[0] + ".npy"
    temp = "./vocaset/templates/FLAME_sample.ply"
    out_path = fa_path.split(".")[0]
    audio_fname = args.wav_path

    template = Mesh(filename=temp)
    predicted_vertices_out = np.load(fa_path).reshape(-1, 5023, 3)
    logger.info("Start output sequence meshes...")
    output_sequence_meshes(predicted_vertices_out, template, out_path)

    logger.info("Start render sequence meshes...")
    render_sequence_meshes(audio_fname, predicted_vertices_out, template, out_path, uv_template_fname='',
                           texture_img_fname='')


if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()
