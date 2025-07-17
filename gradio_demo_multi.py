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
import torch.multiprocessing as mp



#declare some path
BG_DIR="/home/notebook/code/personal/S9059881/IC-Light/imgs/bgs/"
FG_DIR="/home/notebook/code/personal/S9059881/batch-face/images/white_yellow_xxx_thr0.9_bsz32/"
OUTPUT_DIR="/home/notebook/code/personal/S9059881/IC-Light/imgs/output/"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
#set the standard size
IMG_WIDTH = 1024
IMG_HEIGHT = 1024


def initialize_pipelines(device, model_path=None):
    """
    初始化 Stable Diffusion 文本到图像和图像到图像管道
    
    参数:
        device: torch.device - 使用的设备 (cuda 或 cpu)
        model_path: str - IC-Light 模型的路径 (可选)
    
    返回:
        t2i_pipe: StableDiffusionPipeline - 文本到图像管道
        i2i_pipe: StableDiffusionImg2ImgPipeline - 图像到图像管道
    """
    # 1. 加载基础模型
    sd15_name = 'stablediffusionapi/realistic-vision-v51'
    
    tokenizer = CLIPTokenizer.from_pretrained(
        sd15_name, 
        subfolder="tokenizer"
    )
    
    text_encoder = CLIPTextModel.from_pretrained(
        sd15_name, 
        subfolder="text_encoder"
    )
    
    vae = AutoencoderKL.from_pretrained(
        sd15_name, 
        subfolder="vae"
    )
    
    unet = UNet2DConditionModel.from_pretrained(
        sd15_name, 
        subfolder="unet"
    )
    
    # 2. 修改 UNet 结构
    with torch.no_grad():
        new_conv_in = torch.nn.Conv2d(
            8, 
            unet.conv_in.out_channels, 
            unet.conv_in.kernel_size, 
            unet.conv_in.stride, 
            unet.conv_in.padding
        )
        new_conv_in.weight.zero_()
        new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)
        new_conv_in.bias = unet.conv_in.bias
        unet.conv_in = new_conv_in
    
    # 保存原始 forward 方法并创建 hook
    unet_original_forward = unet.forward
    
    def hooked_unet_forward(sample, timestep, encoder_hidden_states, **kwargs):
        c_concat = kwargs['cross_attention_kwargs']['concat_conds'].to(sample)
        c_concat = torch.cat([c_concat] * (sample.shape[0] // c_concat.shape[0]), dim=0)
        new_sample = torch.cat([sample, c_concat], dim=1)
        kwargs['cross_attention_kwargs'] = {}
        return unet_original_forward(new_sample, timestep, encoder_hidden_states, **kwargs)
    
    unet.forward = hooked_unet_forward
    
    # 3. 加载 IC-Light 模型
    if not model_path:
        model_path = '/home/notebook/code/personal/S9059881/IC-Light/models/iclight_sd15_fc.safetensors'
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"IC-Light model not found at {model_path}")
    
    sd_offset = sf.load_file(model_path)
    sd_origin = unet.state_dict()
    sd_merged = {k: sd_origin[k] + sd_offset[k] for k in sd_origin.keys()}
    unet.load_state_dict(sd_merged, strict=True)
    
    # 清理内存
    del sd_offset, sd_origin, sd_merged
    
    # 4. 设置设备并加载模型到设备
    text_encoder = text_encoder.to(device=device, dtype=torch.float16)
    vae = vae.to(device=device, dtype=torch.bfloat16)
    unet = unet.to(device=device, dtype=torch.float16)
    
    # 5. 加载背景移除模型
    rmbg = BriaRMBG.from_pretrained("briaai/RMBG-1.4")
    rmbg = rmbg.to(device=device, dtype=torch.float32)
    
    # 6. 使用 SDP 注意力机制
    unet.set_attn_processor(AttnProcessor2_0())
    vae.set_attn_processor(AttnProcessor2_0())
    
    # 7. 初始化调度器
    dpmpp_2m_sde_karras_scheduler = DPMSolverMultistepScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        algorithm_type="sde-dpmsolver++",
        use_karras_sigmas=True,
        steps_offset=1
    )
    
    # 8. 创建管道
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
    
    # 9. 返回管道和背景移除模型
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
    uc = encode_prompt_inner(negative_prompt, tokenizer, device, text_encoder)

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
def process(input_fg, prompt, t2i_pipe, i2i_pipe, device, vae, tokenizer, text_encoder, unet, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, lowres_denoise, bg_source):
    bg_source = BGSource(bg_source)
    input_bg = None

    if bg_source == BGSource.NONE:
        pass
    elif bg_source == BGSource.LEFT:
        gradient = np.linspace(255, 0, image_width)
        image = np.tile(gradient, (image_height, 1))
        input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
    elif bg_source == BGSource.RIGHT:
        gradient = np.linspace(0, 255, image_width)
        image = np.tile(gradient, (image_height, 1))
        input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
    elif bg_source == BGSource.TOP:
        gradient = np.linspace(255, 0, image_height)[:, None]
        image = np.tile(gradient, (1, image_width))
        input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
    elif bg_source == BGSource.BOTTOM:
        gradient = np.linspace(0, 255, image_height)[:, None]
        image = np.tile(gradient, (1, image_width))
        input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
    else:
        raise 'Wrong initial latent!'

    rng = torch.Generator(device=device).manual_seed(int(seed))

    fg = resize_and_center_crop(input_fg, image_width, image_height)

    concat_conds = numpy2pytorch([fg]).to(device=vae.device, dtype=vae.dtype)
    concat_conds = vae.encode(concat_conds).latent_dist.mode() * vae.config.scaling_factor

    conds, unconds = encode_prompt_pair(positive_prompt=prompt + ', ' + a_prompt, negative_prompt=n_prompt , tokenizer = tokenizer, device=device , text_encoder=text_encoder)

    if input_bg is None:
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
    else:
        bg = resize_and_center_crop(input_bg, image_width, image_height)
        bg_latent = numpy2pytorch([bg]).to(device=vae.device, dtype=vae.dtype)
        bg_latent = vae.encode(bg_latent).latent_dist.mode() * vae.config.scaling_factor
        latents = i2i_pipe(
            image=bg_latent,
            strength=lowres_denoise,
            prompt_embeds=conds,
            negative_prompt_embeds=unconds,
            width=image_width,
            height=image_height,
            num_inference_steps=int(round(steps / lowres_denoise)),
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
    concat_conds = numpy2pytorch([fg]).to(device=vae.device, dtype=vae.dtype)
    concat_conds = vae.encode(concat_conds).latent_dist.mode() * vae.config.scaling_factor

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
    outputs = pytorch2numpy(pixels)
    return outputs


@torch.inference_mode()
def process_relight(input_fg, prompt, t2i_pipe, i2i_pipe, rmbg, device, vae, tokenizer, text_encoder, unet, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, lowres_denoise, bg_source):
    input_fg, matting = run_rmbg(rmbg, device, input_fg)
    #生成的是去掉背景的前景图像
    results = process(input_fg, prompt, t2i_pipe, i2i_pipe, device, vae, tokenizer, text_encoder, unet, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, lowres_denoise, bg_source)
    return input_fg, results


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

self_prompts = [
    "detailed face and skin texture, Window sunlight",
    "detailed face and skin texture, Ocean sunset",
    "detailed face and skin texture, Golden hour glow", 
    "detailed face and skin texture, Natural daylight",
    "detailed face and skin texture, Cozy bedroom lighting",
    "detailed face and skin texture, Window shadow patterns",
    "detailed face and skin texture, Studio softbox lighting",
    "detailed face and skin texture, Homey bedroom illumination",
    "detailed face and skin texture, Dim jazz club lighting",
    "detailed face and skin texture, Sunset fishing boat",
    "detailed face and skin texture, Mediterranean beach light",
    "detailed face and skin texture, Caribbean water reflections",
    "detailed face and skin texture, Greek column sunlight",
    "detailed face and skin texture, Amber yoga studio",
    "detailed face and skin texture, Greenhouse light beams",
    "detailed face and skin texture, Neon grid gradients",
    "detailed face and skin texture, Pink lagoon sunset",
    "detailed face and skin texture, Pixel cityscape",
    "detailed face and skin texture, Sunlit curtain beams",
    "detailed face and skin texture, Neon portrait studio",
    "detailed face and skin texture, Snowy mountain sunlight",
    "detailed face and skin texture, Tropical beach clarity",
    "detailed face and skin texture, Bedside reading lamp",
    "detailed face and skin texture, Golden ocean path",
    "detailed face and skin texture, Fireplace glow",
    "detailed face and skin texture, Urban neon intersection",
    "detailed face and skin texture, Rooftop cocktail lights",
    "detailed face and skin texture, Prairie midday sun",
    "detailed face and skin texture, Vintage desk sunset",
    "detailed face and skin texture, Sunlit library",
    "detailed face and skin texture, Corporate meeting room",
    "detailed face and skin texture, Flower meadow",
    "detailed face and skin texture, Modern dining space",
    "detailed face and skin texture, Traditional Chinese interior",
    "detailed face and skin texture, Campus stadium",
    "detailed face and skin texture, Rainy street reflections",
    "detailed face and skin texture, Desert dusk silhouettes",
    "detailed face and skin texture, Underground metro lighting",
    "detailed face and skin texture, Morning fog forest",
    "detailed face and skin texture, Industrial warehouse skylight",
    "detailed face and skin texture, Greenhouse sunrise",
    "detailed face and skin texture, Subway station fluorescence",
    "detailed face and skin texture, Night market lanterns",
    "detailed face and skin texture, Library reading lamps",
    "detailed face and skin texture, Lighthouse beam rotation",
    "detailed face and skin texture, Airport runway lights",
    "detailed face and skin texture, Candlelit cathedral",
    "detailed face and skin texture, Underwater coral glow"
    # 新增暗光场景扩展 (52条)
    "detailed face and skin texture, Blue hour city balcony",
    "detailed face and skin texture, Midnight library desk lamp",
    "detailed face and skin texture, Jazz bar spotlight contour",
    "detailed face and skin texture, Candlelit dinner glow",
    "detailed face and skin texture, Neon alleyway reflections",
    "detailed face and skin texture, Theater marquee glow",
    "detailed face and skin texture, Aquarium blue illumination",
    "detailed face and skin texture, Fire escape stairwell light",
    "detailed face and skin texture, Noir detective office",
    "detailed face and skin texture, Subway platform fluorescence",
    "detailed face and skin texture, Concert stage backlight",
    "detailed face and skin texture, Rain-streaked car window",
    "detailed face and skin texture, Observatory red light",
    "detailed face and skin texture, Wine cellar ambiance",
    "detailed face and skin texture, Art gallery accent light",
    "detailed face and skin texture, Karaoke booth neon",
    "detailed face and skin texture, Bakery display case",
    "detailed face and skin texture, Vinyl record store",
    "detailed face and skin texture, Darkroom safelight",
    "detailed face and skin texture, Casino slot glow",
    "detailed face and skin texture, Tattoo parlor sign",
    "detailed face and skin texture, Night flower market",
    "detailed face and skin texture, Rooftop pool edge",
    "detailed face and skin texture, Bookstore nook lamp",
    "detailed face and skin texture, Dashboard glow drive",
    "detailed face and skin texture, Fridge light midnight",
    "detailed face and skin texture, Phone screen illumination",
    "detailed face and skin texture, Campfire ember light",
    "detailed face and skin texture, Lightning flash moment",
    "detailed face and skin texture, Police siren reflections",
    "detailed face and skin texture, Fireworks burst light",
    "detailed face and skin texture, Bioluminescent shore",
    "detailed face and skin texture, Northern lights glow",
    "detailed face and skin texture, Welding spark instant",
    "detailed face and skin texture, Lighthouse beam pass",
    "detailed face and skin texture, Hospital monitor",
    "detailed face and skin texture, Dark ride lighting",
    "detailed face and skin texture, Dive torch beam",
    "detailed face and skin texture, Blacklight glow",
    "detailed face and skin texture, Neon repair work",
    "detailed face and skin texture, Concert wristband",
    "detailed face and skin texture, Airplane cabin night",
    "detailed face and skin texture, Photobooth flash",
    "detailed face and skin texture, Mirror ball specks",
    "detailed face and skin texture, Security camera",
    "detailed face and skin texture, Elevator buttons",
    "detailed face and skin texture, ATM screen glow",
    "detailed face and skin texture, Vending machine",
    "detailed face and skin texture, Night construction",
    "detailed face and skin texture, Tokyo alley neon",
    "detailed face and skin texture, Marrakech lanterns",
    "detailed face and skin texture, Venetian mask shop",
    "detailed face and skin texture, Parisian cabaret",
    "detailed face and skin texture, Istanbul oil lamps",
    "detailed face and skin texture, New Orleans jazz"
]

quick_prompts = [[x] for x in quick_prompts]


quick_subjects = [
    'beautiful woman, detailed face',
    'handsome man, detailed face',
]
quick_subjects = [[x] for x in quick_subjects]

class BGSource(Enum):
    NONE = "None"
    LEFT = "Left Light"
    RIGHT = "Right Light"
    TOP = "Top Light"
    BOTTOM = "Bottom Light"

def run_generate_images(gpu, gpu_size, seed):
    generate_images(seed=seed, gpu=gpu, part=gpu, n_parts=gpu_size)

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
    fg_paths.sort()
    
    for ext in bg_extensions:
        bg_paths.extend(glob.glob(os.path.join(BG_DIR,ext)))
    if not fg_paths or not bg_paths:
        logging.error("No input image found, please add more samploe images.")
    # 分配任务
    fg_paths = fg_paths[part::n_parts]
    logging.info(f"gpu{gpu}:load {len(fg_paths)}foreground images({part}/{n_parts})")
    #处理单个照片的重光照
    success_count=0
    LIGHT_DIRECTION=[BGSource.NONE,BGSource.LEFT,BGSource.RIGHT,BGSource.TOP,BGSource.BOTTOM]
    
    total_pairs= len(fg_paths)*10
    logging.info(f"Total {total_pairs} image pairs to process.")
    progress = tqdm(total=total_pairs, desc=f"GPU {gpu}: Processing images")

    for fg_path in fg_paths:
        fg_name = os.path.splitext(os.path.basename(fg_path))[0]
        # for bg_path in bg_paths:
        #     bg_name = os.path.splitext(os.path.basename(bg_path))[0]
        input_fg = np.array(Image.open(fg_path).convert("RGB"))#转换为np数组
        select_prompts = random.sample(self_prompts,k=10)
        for j,me_prompt in enumerate(select_prompts):
        # for light_dir in LIGHT_DIRECTION:
            light_dir = random.choice(list(BGSource))
            output_name = f"{fg_name}_relight_{light_dir.value}_prompt{me_prompt}.png"
            output_path=os.path.join(OUTPUT_DIR,output_name)
            output_fg,results = process_relight(
                input_fg=input_fg,  # 前景图片的 numpy 数组
                prompt=me_prompt,  # Prompt 文本
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
                steps=25,
                a_prompt='best quality',
                n_prompt='lowres, bad anatomy, bad hands, cropped, worst quality',
                cfg=2,
                highres_scale=1.5,
                highres_denoise=0.5,
                lowres_denoise=0.9,
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
    # result=[fg_path,"sunshine from window",light_dir,IMG_WIDTH,IMG_HEIGHT,seed,output_path]
    progress.close()
    logging.info(f"GPU {gpu}: Progress completed. {success_count}/{total_pairs} pairs succeeded.")
    logging.info(f"Results are saved in {OUTPUT_DIR}")

# 主函数
def main():
    """主流程：加载模型并批量处理图片"""
    logging.info("Starting iclight image relighting batch processing...")
    gpu_size = torch.cuda.device_count()
    logging.info(f"Using {gpu_size} GPUS for processing...")

    # 按 GPU 分片任务
    n_parts = gpu_size
    gpu = 7
    generate_images(
        seed=12345,
        gpu=gpu,
        part=gpu,
        n_parts=n_parts,
    )
    # gpu = 1
    # generate_images(
    #     seed=12345,
    #     gpu=gpu,
    #     part=gpu,
    #     n_parts=n_parts,
    # )
    # gpu = 2
    # generate_images(
    #     seed=12345,
    #     gpu=gpu,
    #     part=gpu,
    #     n_parts=n_parts,
    # )
    # gpu = 3
    # generate_images(
    #     seed=12345,
    #     gpu=gpu,
    #     part=gpu,
    #     n_parts=n_parts,
    # )
    # gpu = 4
    # generate_images(
    #     seed=12345,
    #     gpu=gpu,
    #     part=gpu,
    #     n_parts=n_parts,
    # )
    # gpu = 5
    # generate_images(
    #     seed=12345,
    #     gpu=gpu,
    #     part=gpu,
    #     n_parts=n_parts,
    # )
    # gpu = 6
    # generate_images(
    #     seed=12345,
    #     gpu=gpu,
    #     part=gpu,
    #     n_parts=n_parts,
    # )
    # gpu = 7
    # generate_images(
    #     seed=12345,
    #     gpu=gpu,
    #     part=gpu,
    #     n_parts=n_parts,
    # )
    
    # mp.spawn(run_generate_images,args = [gpu_size, seed], nprocs=gpu_size, join=True)
    logging.info("All tasks completed.")



if __name__ == "__main__":
    main()
