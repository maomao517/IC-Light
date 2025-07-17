import os
import re
import shutil
from pathlib import Path

def ensure_trailing_slash(path):
    """确保路径以斜杠结尾，兼容不同操作系统"""
    return os.path.join(path, '')

def get_relight_files(refer_dir, sample_rate=10):
    """从参考目录中获取符合模式的文件，并按采样率筛选"""
    relight_files = []
    pattern = re.compile(r'(.+?)_relight_.*')
    
    for i, entry in enumerate(os.scandir(refer_dir), 1):
        if entry.is_file() and i % sample_rate == 0:
            # 提取文件名（不含扩展名）
            file_name = os.path.splitext(entry.name)[0]
            match = pattern.match(file_name)
            
            if match:
                # 重构文件名，保留前缀并添加.jpg扩展名
                new_name = f"{match.group(1)}.jpg"
                relight_files.append(new_name)
    
    return relight_files

def move_matching_images(input_dir, output_dir, refer_dir, sample_rate=10):
    """
    从参考目录获取文件名模式，在输入目录中查找匹配文件并移动到输出目录
    
    参数:
        input_dir: 源图像目录
        output_dir: 目标目录
        refer_dir: 参考目录，用于提取文件名模式
        sample_rate: 采样率，每N个文件处理1个
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取符合条件的文件名列表
    relight_files = get_relight_files(refer_dir, sample_rate)
    print(len(relight_files))
    # 构建输入目录中所有文件的集合（用于快速查找）
    input_files = {f.name for f in os.scandir(input_dir) if f.is_file()}
    
    # 记录移动的文件数量
    moved_count = 0
    
    for filename in relight_files:
        if filename in input_files:
            src_path = os.path.join(input_dir, filename)
            dst_path = os.path.join(output_dir, filename)
            
            try:
                # 使用shutil.move替代os.system，提供更好的错误处理
                shutil.move(src_path, dst_path)
                moved_count += 1
                print(f"已移动: {filename}")
            except Exception as e:
                print(f"无法移动 {filename}: {e}")
    
    print(f"处理完成: 共移动 {moved_count} 个文件")

if __name__ == "__main__":
    input_dir = "/home/notebook/code/personal/S9059881/batch-face/images/white_yellow_xxx_thr0.9_bsz32"
    output_dir = "/home/notebook/code/personal/S9059881/batch-face/images/white_yellow"
    refer_dir = "/home/notebook/code/personal/S9059881/IC-Light/imgs/output"
    
    # 规范化目录路径
    input_dir = ensure_trailing_slash(input_dir)
    output_dir = ensure_trailing_slash(output_dir)
    refer_dir = ensure_trailing_slash(refer_dir)
    
    # 执行文件移动操作，每10个文件处理1个
    move_matching_images(input_dir, output_dir, refer_dir, sample_rate=10)