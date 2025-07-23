import os
import logging
import glob
from tqdm import tqdm
import math
import random
import gradio as gr
import numpy as np
import torch
import safetensors.torch as sf
import db_examples
from huggingface_hub import hf_hub_download
from PIL import Image
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler, EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler
from diffusers.models.attention_processor import AttnProcessor2_0
from transformers import CLIPTextModel, CLIPTokenizer
from briarmbg import BriaRMBG
from enum import Enum
from torch.hub import download_url_to_file
import argparse

#declare some path
BG_DIR="/home/notebook/code/personal/S9059881/IC-Light/imgs/bgs/"
FG_DIR="/home/notebook/code/personal/S9059881/batch-face/images/white_yellow_xxx_thr0.9_bsz32/"
OUTPUT_DIR="/home/notebook/code/personal/S9059881/IC-Light/imgs/output_bg3/"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
#set the standard size
IMG_WIDTH = 512
IMG_HEIGHT =640


#使用方法：python gradio_demo_multi.py --gpu 0
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, required=True, help='gpu number to be used')
args = parser.parse_args()

def initialize_pipelines(device, model_path=None):
    # 'stablediffusionapi/realistic-vision-v51'
    # 'runwayml/stable-diffusion-v1-5'
    sd15_name = 'stablediffusionapi/realistic-vision-v51'
    tokenizer = CLIPTokenizer.from_pretrained(sd15_name, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(sd15_name, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(sd15_name, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(sd15_name, subfolder="unet")
    rmbg = BriaRMBG.from_pretrained("briaai/RMBG-1.4")



    # Change UNet

    with torch.no_grad():
        new_conv_in = torch.nn.Conv2d(12, unet.conv_in.out_channels, unet.conv_in.kernel_size, unet.conv_in.stride, unet.conv_in.padding)
        new_conv_in.weight.zero_()
        new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)
        new_conv_in.bias = unet.conv_in.bias
        unet.conv_in = new_conv_in

    unet_original_forward = unet.forward


    def hooked_unet_forward(sample, timestep, encoder_hidden_states, **kwargs):
        c_concat = kwargs['cross_attention_kwargs']['concat_conds'].to(sample)
        c_concat = torch.cat([c_concat] * (sample.shape[0] // c_concat.shape[0]), dim=0)
        new_sample = torch.cat([sample, c_concat], dim=1)
        kwargs['cross_attention_kwargs'] = {}
        return unet_original_forward(new_sample, timestep, encoder_hidden_states, **kwargs)


    unet.forward = hooked_unet_forward

    # Load
    if not model_path:
        model_path = '/home/notebook/code/personal/S9059881/IC-Light/models/iclight_sd15_fbc.safetensors'
    # use "wget https://hf-mirror.com/lllyasviel/ic-light/resolve/main/iclight_sd15_fbc.safetensors" to replace the download_url_to_path
    if not os.path.exists(model_path):
        raise RuntimeError(f"模型未下载到{model_path}路径")
    sd_offset = sf.load_file(model_path)
    sd_origin = unet.state_dict()
    keys = sd_origin.keys()
    sd_merged = {k: sd_origin[k] + sd_offset[k] for k in sd_origin.keys()}
    unet.load_state_dict(sd_merged, strict=True)
    del sd_offset, sd_origin, sd_merged, keys

    # Device
    text_encoder = text_encoder.to(device=device, dtype=torch.float16)
    vae = vae.to(device=device, dtype=torch.bfloat16)
    unet = unet.to(device=device, dtype=torch.float16)
    rmbg = rmbg.to(device=device, dtype=torch.float32)

    # SDP

    unet.set_attn_processor(AttnProcessor2_0())
    vae.set_attn_processor(AttnProcessor2_0())

    # Samplers

    ddim_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )

    euler_a_scheduler = EulerAncestralDiscreteScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        steps_offset=1
    )

    dpmpp_2m_sde_karras_scheduler = DPMSolverMultistepScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        algorithm_type="sde-dpmsolver++",
        use_karras_sigmas=True,
        steps_offset=1
    )

    # Pipelines

    t2i_pipe = StableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=dpmpp_2m_sde_karras_scheduler,
        safety_checker=None,
        requires_safety_checker=False,
        feature_extractor=None,
        image_encoder=None
    )

    i2i_pipe = StableDiffusionImg2ImgPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=dpmpp_2m_sde_karras_scheduler,
        safety_checker=None,
        requires_safety_checker=False,
        feature_extractor=None,
        image_encoder=None
    )
    return t2i_pipe, i2i_pipe, rmbg, vae, tokenizer, text_encoder, unet

@torch.inference_mode()
def encode_prompt_inner(txt: str, tokenizer, device, text_encoder):
    max_length = tokenizer.model_max_length
    chunk_length = tokenizer.model_max_length - 2
    id_start = tokenizer.bos_token_id
    id_end = tokenizer.eos_token_id
    id_pad = id_end

    def pad(x, p, i):
        return x[:i] if len(x) >= i else x + [p] * (i - len(x))

    tokens = tokenizer(txt, truncation=False, add_special_tokens=False)["input_ids"]
    chunks = [[id_start] + tokens[i: i + chunk_length] + [id_end] for i in range(0, len(tokens), chunk_length)]
    chunks = [pad(ck, id_pad, max_length) for ck in chunks]

    token_ids = torch.tensor(chunks).to(device=device, dtype=torch.int64)
    conds = text_encoder(token_ids).last_hidden_state

    return conds


@torch.inference_mode()
def encode_prompt_pair(positive_prompt, negative_prompt, tokenizer, device, text_encoder):
    c = encode_prompt_inner(positive_prompt, tokenizer, device, text_encoder)
    uc = encode_prompt_inner(negative_prompt,  tokenizer, device, text_encoder)

    c_len = float(len(c))
    uc_len = float(len(uc))
    max_count = max(c_len, uc_len)
    c_repeat = int(math.ceil(max_count / c_len))
    uc_repeat = int(math.ceil(max_count / uc_len))
    max_chunk = max(len(c), len(uc))

    c = torch.cat([c] * c_repeat, dim=0)[:max_chunk]
    uc = torch.cat([uc] * uc_repeat, dim=0)[:max_chunk]

    c = torch.cat([p[None, ...] for p in c], dim=1)
    uc = torch.cat([p[None, ...] for p in uc], dim=1)

    return c, uc


@torch.inference_mode()
def pytorch2numpy(imgs, quant=True):
    results = []
    for x in imgs:
        y = x.movedim(0, -1)

        if quant:
            y = y * 127.5 + 127.5
            y = y.detach().float().cpu().numpy().clip(0, 255).astype(np.uint8)
        else:
            y = y * 0.5 + 0.5
            y = y.detach().float().cpu().numpy().clip(0, 1).astype(np.float32)

        results.append(y)
    return results


@torch.inference_mode()
def numpy2pytorch(imgs):
    h = torch.from_numpy(np.stack(imgs, axis=0)).float() / 127.0 - 1.0  # so that 127 must be strictly 0.0
    h = h.movedim(-1, 1)
    return h


def resize_and_center_crop(image, target_width, target_height):
    pil_image = Image.fromarray(image)
    original_width, original_height = pil_image.size
    scale_factor = max(target_width / original_width, target_height / original_height)
    resized_width = int(round(original_width * scale_factor))
    resized_height = int(round(original_height * scale_factor))
    resized_image = pil_image.resize((resized_width, resized_height), Image.LANCZOS)
    left = (resized_width - target_width) / 2
    top = (resized_height - target_height) / 2
    right = (resized_width + target_width) / 2
    bottom = (resized_height + target_height) / 2
    cropped_image = resized_image.crop((left, top, right, bottom))
    return np.array(cropped_image)


def resize_without_crop(image, target_width, target_height):
    pil_image = Image.fromarray(image)
    resized_image = pil_image.resize((target_width, target_height), Image.LANCZOS)
    return np.array(resized_image)


@torch.inference_mode()
def run_rmbg(rmbg, device, img, sigma=0.0):
    H, W, C = img.shape
    assert C == 3
    k = (256.0 / float(H * W)) ** 0.5
    feed = resize_without_crop(img, int(64 * round(W * k)), int(64 * round(H * k)))
    feed = numpy2pytorch([feed]).to(device=device, dtype=torch.float32)
    alpha = rmbg(feed)[0][0]
    alpha = torch.nn.functional.interpolate(alpha, size=(H, W), mode="bilinear")
    alpha = alpha.movedim(1, -1)[0]
    alpha = alpha.detach().float().cpu().numpy().clip(0, 1)
    result = 127 + (img.astype(np.float32) - 127 + sigma) * alpha
    return result.clip(0, 255).astype(np.uint8), alpha


@torch.inference_mode()
def process(input_fg, input_bg, prompt, t2i_pipe, i2i_pipe, device, vae, tokenizer, text_encoder, unet, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, bg_source):
    bg_source = BGSource(bg_source)

    if bg_source == BGSource.UPLOAD:
        pass
    elif bg_source == BGSource.UPLOAD_FLIP:
        input_bg = np.fliplr(input_bg)
    elif bg_source == BGSource.GREY:
        input_bg = np.zeros(shape=(image_height, image_width, 3), dtype=np.uint8) + 64
    elif bg_source == BGSource.LEFT:
        gradient = np.linspace(224, 32, image_width)
        image = np.tile(gradient, (image_height, 1))
        input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
    elif bg_source == BGSource.RIGHT:
        gradient = np.linspace(32, 224, image_width)
        image = np.tile(gradient, (image_height, 1))
        input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
    elif bg_source == BGSource.TOP:
        gradient = np.linspace(224, 32, image_height)[:, None]
        image = np.tile(gradient, (1, image_width))
        input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
    elif bg_source == BGSource.BOTTOM:
        gradient = np.linspace(32, 224, image_height)[:, None]
        image = np.tile(gradient, (1, image_width))
        input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
    else:
        raise 'Wrong background source!'

    rng = torch.Generator(device=device).manual_seed(seed)

    fg = resize_and_center_crop(input_fg, image_width, image_height)
    bg = resize_and_center_crop(input_bg, image_width, image_height)
    concat_conds = numpy2pytorch([fg, bg]).to(device=vae.device, dtype=vae.dtype)
    concat_conds = vae.encode(concat_conds).latent_dist.mode() * vae.config.scaling_factor
    concat_conds = torch.cat([c[None, ...] for c in concat_conds], dim=1)

    conds, unconds = encode_prompt_pair(positive_prompt=prompt + ', ' + a_prompt, negative_prompt=n_prompt, tokenizer = tokenizer, device=device , text_encoder=text_encoder)

    latents = t2i_pipe(
        prompt_embeds=conds,
        negative_prompt_embeds=unconds,
        width=image_width,
        height=image_height,
        num_inference_steps=steps,
        num_images_per_prompt=num_samples,
        generator=rng,
        output_type='latent',
        guidance_scale=cfg,
        cross_attention_kwargs={'concat_conds': concat_conds},
    ).images.to(vae.dtype) / vae.config.scaling_factor

    pixels = vae.decode(latents).sample
    pixels = pytorch2numpy(pixels)
    pixels = [resize_without_crop(
        image=p,
        target_width=int(round(image_width * highres_scale / 64.0) * 64),
        target_height=int(round(image_height * highres_scale / 64.0) * 64))
    for p in pixels]

    pixels = numpy2pytorch(pixels).to(device=vae.device, dtype=vae.dtype)
    latents = vae.encode(pixels).latent_dist.mode() * vae.config.scaling_factor
    latents = latents.to(device=unet.device, dtype=unet.dtype)

    image_height, image_width = latents.shape[2] * 8, latents.shape[3] * 8
    fg = resize_and_center_crop(input_fg, image_width, image_height)
    bg = resize_and_center_crop(input_bg, image_width, image_height)
    concat_conds = numpy2pytorch([fg, bg]).to(device=vae.device, dtype=vae.dtype)
    concat_conds = vae.encode(concat_conds).latent_dist.mode() * vae.config.scaling_factor
    concat_conds = torch.cat([c[None, ...] for c in concat_conds], dim=1)

    latents = i2i_pipe(
        image=latents,
        strength=highres_denoise,
        prompt_embeds=conds,
        negative_prompt_embeds=unconds,
        width=image_width,
        height=image_height,
        num_inference_steps=int(round(steps / highres_denoise)),
        num_images_per_prompt=num_samples,
        generator=rng,
        output_type='latent',
        guidance_scale=cfg,
        cross_attention_kwargs={'concat_conds': concat_conds},
    ).images.to(vae.dtype) / vae.config.scaling_factor

    pixels = vae.decode(latents).sample
    pixels = pytorch2numpy(pixels, quant=False)

    return pixels, [fg, bg]


@torch.inference_mode()
def process_relight(input_fg, input_bg, prompt, t2i_pipe, i2i_pipe, rmbg, device, vae, tokenizer, text_encoder, unet, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, bg_source):
    input_fg, matting = run_rmbg(rmbg, device, input_fg)
    results, extra_images = process(input_fg, input_bg, prompt, t2i_pipe, i2i_pipe, device, vae, tokenizer, text_encoder, unet, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, bg_source)
    results = [(x * 255.0).clip(0, 255).astype(np.uint8) for x in results]
    return results + extra_images


quick_prompts = [
    'sunshine from window',
    'neon light, city',
    'sunset over sea',
    'golden time',
    'sci-fi RGB glowing, cyberpunk',
    'natural lighting',
    'warm atmosphere, at home, bedroom',
    'magic lit',
    'evil, gothic, Yharnam',
    'light and shadow',
    'shadow from window',
    'soft studio lighting',
    'home atmosphere, cozy bedroom illumination',
    'neon, Wong Kar-wai, warm',
    'Sunset over sea',
    'Chicago jazz club with dim lights',
    'An abstract, digital-art inspired backdrop',
    'Bright, saturated, clear-sky scene, reflecting distinctive facial details.',
    'Vibrant neon-lit night scene, neutralizing reflections on facial features.',
    'Timeless, classic architectural backdrop, reflecting well-defined facial features.',
    'Serene blue sea, capturing clear facial dimensions and reflections.',
    'Warm, cozy room backdrop, resonating well-modeled facial attributes.',
    'Snowy setting with crisp cold reflections, highlighting sharp facial features.',
    'Contrast of vibrant rainforest canopy, presenting clear facial structure and reflections.',
    'Energetic market ambiance, showcasing distinguished facial details.',
    'Under the stars, focusing on facial structures and natural reflections.',
]

quick_prompts = [[x] for x in quick_prompts]


quick_subjects = [
    'beautiful woman, detailed face',
    'handsome man, detailed face',
]
quick_subjects = [[x] for x in quick_subjects]

class BGSource(Enum):
    UPLOAD = "Use Background Image"
    UPLOAD_FLIP = "Use Flipped Background Image"
    LEFT = "Left Light"
    RIGHT = "Right Light"
    TOP = "Top Light"
    BOTTOM = "Bottom Light"
    GREY = "Ambient"


def generate_images(seed:int=12345, gpu:int=0, part:int=0, n_parts:int=1):
    device = torch.device(f'cuda:{gpu}')
    t2i_pipe, i2i_pipe, rmbg, vae, tokenizer,text_encoder, unet = initialize_pipelines(device)
    result=[]
    fg_extensions = ["*.jpg","*.png","*.jpeg","*.bmp","*.webp"]
    bg_extensions = ["*.jpg","*.png","*.jpeg","*.bmp","*.webp"]
    fg_paths=[]
    bg_paths=[]
    for ext in fg_extensions:
        fg_paths.extend(glob.glob(os.path.join(FG_DIR,ext)))
    for ext in bg_extensions:
        bg_paths.extend(glob.glob(os.path.join(BG_DIR,ext)))
    if not fg_paths or not bg_paths:
        logging.error("No input image found, please add more samploe images.")
    logging.info(f"Found {len(fg_paths)} foreground images and {len(bg_paths)} background images")
    
    # 分配任务
    fg_paths = fg_paths[part::n_parts]
    logging.info(f"gpu{gpu}:load {len(fg_paths)}foreground images({part}/{n_parts})")

    #处理单个照片的重光照
    success_count=0
    LIGHT_DIRECTION=[BGSource.UPLOAD, BGSource.UPLOAD_FLIP, BGSource.LEFT,BGSource.RIGHT,BGSource.TOP,BGSource.BOTTOM, BGSource.GREY]
    total_pairs= len(fg_paths)*10
    logging.info(f"Total {total_pairs} image pairs to process.")
    progress = tqdm(total=total_pairs,desc="processing images")

    for fg_path in fg_paths:
        fg_name = os.path.splitext(os.path.basename(fg_path))[0]
        # for bg_path in bg_paths:
        #     bg_name = os.path.splitext(os.path.basename(bg_path))[0]
        input_fg = np.array(Image.open(fg_path).convert("RGB"))#转换为np数组
        select_bg_paths = random.sample(bg_paths,k=10)
        for bg_path in select_bg_paths:
            input_bg = np.array(Image.open(bg_path).convert("RGB"))#转换为np数组
            bg_name = os.path.splitext(os.path.basename(bg_path))[0]
        # for light_dir in LIGHT_DIRECTION:
            light_dir = BGSource.UPLOAD #默认使用背景图片
            output_name = f"{fg_name}_relight_{light_dir.value}_{bg_name}.png"
            output_path=os.path.join(OUTPUT_DIR,output_name)
            results = process_relight(
                input_fg=input_fg,  # 前景图片的 numpy 数组
                input_bg=input_bg,
                prompt="natural face, smooth skin, soft natural lighting, no overexposure, seamless blend with background",
                t2i_pipe=t2i_pipe,
                i2i_pipe=i2i_pipe,
                rmbg=rmbg,
                device=device,
                vae=vae,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                unet=unet,
                image_width=IMG_WIDTH,
                image_height=IMG_HEIGHT,
                num_samples=1,
                seed=seed,
                steps=35,
                a_prompt="best quality, soft shadows, balanced light, cinematic skin texture, subtle skin pores, natural oil sheen",  # 正向增强柔和光照
                n_prompt="overexposed, bright spots, harsh light, grainy skin, wax figure, airbrushed skin, perfect skin",  # 排除过曝和硬光
                cfg=3.2,
                highres_scale=1.2,
                highres_denoise=0.6,
                bg_source=light_dir.value,
            )
            if len(results) > 0:
                # images=convert_to_image(results[0])
                Image.fromarray(results[0]).save(output_path)
                success_count+=1
            else :
                print("error happens")
                return
            progress.update(1)

    progress.close()
    logging.info(f"Progress completed.{success_count}/{total_pairs}pairs succeed.")
    logging.info(f"Results are saved in {OUTPUT_DIR}")

# 主函数
def main():
    """主流程：加载模型并批量处理图片"""
    logging.info("Starting iclight image relighting batch processing...")
    gpu_size = torch.cuda.device_count()
    logging.info(f"Using {gpu_size} GPUS for processing...")

    # 按 GPU 分片任务
    n_parts = gpu_size
    # 使用tmux开多个窗口，实现多卡推理
    gpu = args.gpu
    generate_images(
        seed=12345,
        gpu=gpu,
        part=gpu,
        n_parts=n_parts,
    )
    logging.info("All tasks completed.")

if __name__ == "__main__":
    main()


# block = gr.Blocks().queue()
# with block:
#     with gr.Row():
#         gr.Markdown("## IC-Light (Relighting with Foreground and Background Condition)")
#     with gr.Row():
#         with gr.Column():
#             with gr.Row():
#                 input_fg = gr.Image(source='upload', type="numpy", label="Foreground", height=480)
#                 input_bg = gr.Image(source='upload', type="numpy", label="Background", height=480)
#             prompt = gr.Textbox(label="Prompt")
#             bg_source = gr.Radio(choices=[e.value for e in BGSource],
#                                  value=BGSource.UPLOAD.value,
#                                  label="Background Source", type='value')

#             example_prompts = gr.Dataset(samples=quick_prompts, label='Prompt Quick List', components=[prompt])
#             bg_gallery = gr.Gallery(height=450, object_fit='contain', label='Background Quick List', value=db_examples.bg_samples, columns=5, allow_preview=False)
#             relight_button = gr.Button(value="Relight")

#             with gr.Group():
#                 with gr.Row():
#                     num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
#                     seed = gr.Number(label="Seed", value=12345, precision=0)
#                 with gr.Row():
#                     image_width = gr.Slider(label="Image Width", minimum=256, maximum=1024, value=512, step=64)
#                     image_height = gr.Slider(label="Image Height", minimum=256, maximum=1024, value=640, step=64)

#             with gr.Accordion("Advanced options", open=False):
#                 steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
#                 cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=32.0, value=7.0, step=0.01)
#                 highres_scale = gr.Slider(label="Highres Scale", minimum=1.0, maximum=3.0, value=1.5, step=0.01)
#                 highres_denoise = gr.Slider(label="Highres Denoise", minimum=0.1, maximum=0.9, value=0.5, step=0.01)
#                 a_prompt = gr.Textbox(label="Added Prompt", value='best quality')
#                 n_prompt = gr.Textbox(label="Negative Prompt",
#                                       value='lowres, bad anatomy, bad hands, cropped, worst quality')
#                 normal_button = gr.Button(value="Compute Normal (4x Slower)")
#         with gr.Column():
#             result_gallery = gr.Gallery(height=832, object_fit='contain', label='Outputs')
#     with gr.Row():
#         dummy_image_for_outputs = gr.Image(visible=False, label='Result')
#         gr.Examples(
#             fn=lambda *args: [args[-1]],
#             examples=db_examples.background_conditioned_examples,
#             inputs=[
#                 input_fg, input_bg, prompt, bg_source, image_width, image_height, seed, dummy_image_for_outputs
#             ],
#             outputs=[result_gallery],
#             run_on_click=True, examples_per_page=1024
#         )
#     ips = [input_fg, input_bg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, bg_source]
#     relight_button.click(fn=process_relight, inputs=ips, outputs=[result_gallery])
#     normal_button.click(fn=process_normal, inputs=ips, outputs=[result_gallery])
#     example_prompts.click(lambda x: x[0], inputs=example_prompts, outputs=prompt, show_progress=False, queue=False)

#     def bg_gallery_selected(gal, evt: gr.SelectData):
#         return gal[evt.index]['name']

#     bg_gallery.select(bg_gallery_selected, inputs=bg_gallery, outputs=input_bg)


# block.launch(server_name='0.0.0.0')