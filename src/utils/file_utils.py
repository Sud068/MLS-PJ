"""
文件操作工具
提供文件系统操作的辅助函数
"""

import os
import shutil
import glob
import hashlib
import tempfile
import zipfile
import tarfile
import logging
from typing import List, Tuple, Optional, Dict, Any,Union
from .validation import validate_not_none, validate_type, validate_file_path

logger = logging.getLogger(__name__)


def ensure_dir_exists(dir_path: str) -> str:
    """
    确保目录存在，如果不存在则创建

    参数:
    dir_path: 目录路径

    返回:
    创建的目录路径
    """
    validate_not_none(dir_path, "dir_path")
    validate_type(dir_path, str, "dir_path")

    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")

    return dir_path


def safe_remove(path: str) -> bool:
    """
    安全删除文件或目录

    参数:
    path: 要删除的路径

    返回:
    是否成功删除
    """
    try:
        if os.path.isfile(path):
            os.remove(path)
            logger.debug(f"Removed file: {path}")
            return True
        elif os.path.isdir(path):
            shutil.rmtree(path)
            logger.debug(f"Removed directory: {path}")
            return True
    except Exception as e:
        logger.error(f"Error removing {path}: {str(e)}")

    return False


def find_files(pattern: str, root_dir: str = '.', recursive: bool = True) -> List[str]:
    """
    查找匹配模式的文件

    参数:
    pattern: 文件模式 (如 '*.txt')
    root_dir: 搜索的根目录
    recursive: 是否递归搜索

    返回:
    匹配的文件路径列表
    """
    root_dir = ensure_dir_exists(root_dir)
    if recursive:
        pattern = os.path.join(root_dir, '**', pattern)
    else:
        pattern = os.path.join(root_dir, pattern)

    return glob.glob(pattern, recursive=recursive)


def get_file_hash(file_path: str, algorithm: str = 'sha256') -> str:
    """
    计算文件的哈希值

    参数:
    file_path: 文件路径
    algorithm: 哈希算法 (md5, sha1, sha256, sha512)

    返回:
    文件的哈希值
    """
    validate_file_path(file_path, must_exist=True)

    hash_func = getattr(hashlib, algorithm, hashlib.sha256)
    buffer_size = 65536  # 64KB块

    with open(file_path, 'rb') as f:
        file_hash = hash_func()
        while chunk := f.read(buffer_size):
            file_hash.update(chunk)

    return file_hash.hexdigest()


def compare_files(file1: str, file2: str) -> bool:
    """
    比较两个文件是否相同

    参数:
    file1: 第一个文件路径
    file2: 第二个文件路径

    返回:
    文件是否相同
    """
    validate_file_path(file1, must_exist=True)
    validate_file_path(file2, must_exist=True)

    # 首先比较文件大小
    if os.path.getsize(file1) != os.path.getsize(file2):
        return False

    # 比较文件内容
    with open(file1, 'rb') as f1, open(file2, 'rb') as f2:
        while True:
            chunk1 = f1.read(4096)
            chunk2 = f2.read(4096)
            if chunk1 != chunk2:
                return False
            if not chunk1:
                break

    return True


def create_temp_dir(prefix: str = 'xai_') -> str:
    """
    创建临时目录

    参数:
    prefix: 目录名前缀

    返回:
    临时目录路径
    """
    return tempfile.mkdtemp(prefix=prefix)


def create_temp_file(prefix: str = 'xai_', suffix: str = '.tmp') -> str:
    """
    创建临时文件

    参数:
    prefix: 文件名前缀
    suffix: 文件名后缀

    返回:
    临时文件路径
    """
    with tempfile.NamedTemporaryFile(prefix=prefix, suffix=suffix, delete=False) as f:
        return f.name


def zip_directory(dir_path: str, output_path: str) -> str:
    """
    压缩目录为ZIP文件

    参数:
    dir_path: 要压缩的目录
    output_path: 输出的ZIP文件路径

    返回:
    创建的ZIP文件路径
    """
    dir_path = ensure_dir_exists(dir_path)
    output_dir = os.path.dirname(output_path) or '.'
    ensure_dir_exists(output_dir)

    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(dir_path):
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, dir_path)
                zipf.write(file_path, rel_path)

    logger.info(f"Created ZIP archive: {output_path}")
    return output_path


def unzip_file(zip_path: str, output_dir: str) -> str:
    """
    解压ZIP文件到目录

    参数:
    zip_path: ZIP文件路径
    output_dir: 输出目录

    返回:
    解压后的目录路径
    """
    validate_file_path(zip_path, must_exist=True, extension='.zip')
    output_dir = ensure_dir_exists(output_dir)

    with zipfile.ZipFile(zip_path, 'r') as zipf:
        zipf.extractall(output_dir)

    logger.info(f"Extracted ZIP archive to: {output_dir}")
    return output_dir


def compress_to_tar(source_path: str, output_path: str,
                    compression: str = 'gz') -> str:
    """
    压缩文件或目录为tar文件

    参数:
    source_path: 源文件或目录路径
    output_path: 输出tar文件路径
    compression: 压缩算法 (None, 'gz', 'bz2', 'xz')

    返回:
    创建的tar文件路径
    """
    mode_map = {
        None: 'w',
        'gz': 'w:gz',
        'bz2': 'w:bz2',
        'xz': 'w:xz'
    }

    if compression not in mode_map:
        raise ValueError(f"Unsupported compression: {compression}")

    mode = mode_map[compression]

    with tarfile.open(output_path, mode) as tar:
        if os.path.isdir(source_path):
            tar.add(source_path, arcname=os.path.basename(source_path))
        else:
            tar.add(source_path)

    logger.info(f"Created TAR archive: {output_path}")
    return output_path


def decompress_tar(tar_path: str, output_dir: str) -> str:
    """
    解压tar文件到目录

    参数:
    tar_path: tar文件路径
    output_dir: 输出目录

    返回:
    解压后的目录路径
    """
    validate_file_path(tar_path, must_exist=True)
    output_dir = ensure_dir_exists(output_dir)

    with tarfile.open(tar_path, 'r:*') as tar:
        tar.extractall(path=output_dir)

    logger.info(f"Extracted TAR archive to: {output_dir}")
    return output_dir


def read_file_chunks(file_path: str, chunk_size: int = 8192) -> bytes:
    """
    分块读取文件

    参数:
    file_path: 文件路径
    chunk_size: 块大小 (字节)

    返回:
    文件内容的生成器
    """
    validate_file_path(file_path, must_exist=True)

    with open(file_path, 'rb') as f:
        while chunk := f.read(chunk_size):
            yield chunk


def copy_file(src: str, dst: str, overwrite: bool = False) -> str:
    """
    复制文件

    参数:
    src: 源文件路径
    dst: 目标文件路径
    overwrite: 是否覆盖已存在文件

    返回:
    复制的文件路径
    """
    validate_file_path(src, must_exist=True)

    if os.path.exists(dst) and not overwrite:
        raise FileExistsError(f"File already exists: {dst}")

    ensure_dir_exists(os.path.dirname(dst))
    shutil.copy2(src, dst)
    logger.debug(f"Copied file: {src} -> {dst}")
    return dst


def move_file(src: str, dst: str, overwrite: bool = False) -> str:
    """
    移动文件

    参数:
    src: 源文件路径
    dst: 目标文件路径
    overwrite: 是否覆盖已存在文件

    返回:
    移动后的文件路径
    """
    validate_file_path(src, must_exist=True)

    if os.path.exists(dst) and not overwrite:
        raise FileExistsError(f"File already exists: {dst}")

    ensure_dir_exists(os.path.dirname(dst))
    shutil.move(src, dst)
    logger.debug(f"Moved file: {src} -> {dst}")
    return dst


def get_file_size(file_path: str, human_readable: bool = False) -> Union[int, str]:
    """
    获取文件大小

    参数:
    file_path: 文件路径
    human_readable: 是否返回人类可读格式

    返回:
    文件大小 (字节或人类可读字符串)
    """
    validate_file_path(file_path, must_exist=True)

    size = os.path.getsize(file_path)

    if human_readable:
        # 转换为人类可读格式
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024:
                return f"{size:.2f} {unit}"
            size /= 1024
        return f"{size:.2f} PB"
    else:
        return size