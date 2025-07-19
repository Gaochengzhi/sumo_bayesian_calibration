#!/usr/bin/env python3
import os
import subprocess
import argparse
import sys

def compress_pdf(input_folder, output_folder=None, quality='/ebook'):
    """
    压缩指定文件夹中的所有PDF文件
    
    参数:
    input_folder: 包含PDF文件的输入文件夹路径
    output_folder: 输出文件夹路径，默认为输入文件夹下的compressed子文件夹
    quality: PDF压缩质量，可选值: /screen, /ebook, /printer, /prepress, /default
    """
    # 验证输入文件夹是否存在
    if not os.path.isdir(input_folder):
        print(f"错误: 输入文件夹 '{input_folder}' 不存在")
        return False
    
    # 设置输出文件夹
    if output_folder is None:
        output_folder = os.path.join(input_folder, "compressed")
    
    # 创建输出文件夹（如果不存在）
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 获取所有PDF文件
    pdf_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print(f"警告: 在 '{input_folder}' 中没有找到PDF文件")
        return False
    
    # 处理每个PDF文件
    success_count = 0
    for pdf_file in pdf_files:
        input_path = os.path.join(input_folder, pdf_file)
        
        # 创建输出文件名（保持原始文件名，添加_compressed后缀）
        filename_without_ext = os.path.splitext(pdf_file)[0]
        output_filename = f"{filename_without_ext}_compressed.pdf"
        output_path = os.path.join(output_folder, output_filename)
        
        print(f"正在压缩: {pdf_file}")
        
        # 构建Ghostscript命令
        gs_command = [
            'gs',
            '-sDEVICE=pdfwrite',
            '-dCompatibilityLevel=1.4',
            f'-dPDFSETTINGS={quality}',
            '-dNOPAUSE',
            '-dQUIET',
            '-dBATCH',
            f'-sOutputFile={output_path}',
            input_path
        ]
        
        # 执行命令
        try:
            subprocess.run(gs_command, check=True)
            success_count += 1
            
            # 获取文件大小信息用于显示
            original_size = os.path.getsize(input_path)
            compressed_size = os.path.getsize(output_path)
            reduction = (1 - compressed_size / original_size) * 100
            
            print(f"  原始大小: {original_size/1024:.1f} KB")
            print(f"  压缩大小: {compressed_size/1024:.1f} KB")
            print(f"  减少: {reduction:.1f}%")
            
        except subprocess.CalledProcessError as e:
            print(f"  压缩失败: {e}")
    
    print(f"\n成功压缩 {success_count}/{len(pdf_files)} 个PDF文件")
    print(f"压缩后的文件保存在: {output_folder}")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='批量压缩PDF文件')
    parser.add_argument('input_folder', help='包含PDF文件的输入文件夹路径')
    parser.add_argument('-o', '--output_folder', help='输出文件夹路径（默认为输入文件夹下的compressed子文件夹）')
    parser.add_argument('-q', '--quality', default='/ebook', 
                        choices=['/screen', '/ebook', '/printer', '/prepress', '/default'],
                        help='PDF压缩质量（默认为/ebook）')
    
    args = parser.parse_args()
    
    compress_pdf(args.input_folder, args.output_folder, args.quality)
